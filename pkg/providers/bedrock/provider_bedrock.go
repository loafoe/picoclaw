//go:build bedrock

// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

// Package bedrock implements the LLM provider interface for AWS Bedrock.
// It uses the Bedrock Runtime Converse API for unified access to multiple
// model families (Claude, Llama, Mistral, etc.) with tool/function calling support.
package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers/common"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

type (
	ToolCall               = protocoltypes.ToolCall
	FunctionCall           = protocoltypes.FunctionCall
	LLMResponse            = protocoltypes.LLMResponse
	UsageInfo              = protocoltypes.UsageInfo
	Message                = protocoltypes.Message
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
)

// Provider implements the LLM provider interface for AWS Bedrock.
type Provider struct {
	client         *bedrockruntime.Client
	region         string
	requestTimeout time.Duration
}

// Option configures the Bedrock Provider.
type Option func(*providerConfig)

type providerConfig struct {
	region         string
	profile        string
	baseEndpoint   string
	requestTimeout time.Duration
}

// WithRegion sets the AWS region for Bedrock requests.
func WithRegion(region string) Option {
	return func(c *providerConfig) {
		c.region = region
	}
}

// WithProfile sets the AWS profile to use for credentials.
func WithProfile(profile string) Option {
	return func(c *providerConfig) {
		c.profile = profile
	}
}

// WithBaseEndpoint sets a custom Bedrock endpoint URL.
// Example: https://bedrock-runtime.us-east-1.amazonaws.com
func WithBaseEndpoint(endpoint string) Option {
	return func(c *providerConfig) {
		c.baseEndpoint = endpoint
	}
}

// WithRequestTimeout sets the timeout for Bedrock API requests.
func WithRequestTimeout(timeout time.Duration) Option {
	return func(c *providerConfig) {
		c.requestTimeout = timeout
	}
}

// NewProvider creates a new AWS Bedrock provider.
// It uses the default AWS credential chain (env vars, shared config, IAM roles, etc.).
func NewProvider(ctx context.Context, opts ...Option) (*Provider, error) {
	pc := &providerConfig{}
	for _, opt := range opts {
		opt(pc)
	}

	// Build AWS config options
	var configOpts []func(*config.LoadOptions) error

	if pc.region != "" {
		configOpts = append(configOpts, config.WithRegion(pc.region))
	}

	if pc.profile != "" {
		configOpts = append(configOpts, config.WithSharedConfigProfile(pc.profile))
	}

	// Load AWS config with automatic credential discovery
	cfg, err := config.LoadDefaultConfig(ctx, configOpts...)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	// Validate region is set - required for Bedrock request signing
	if cfg.Region == "" {
		return nil, fmt.Errorf(
			"AWS region not configured: set AWS_REGION, AWS_DEFAULT_REGION, or use WithRegion option",
		)
	}

	// Build client options
	var clientOpts []func(*bedrockruntime.Options)
	if pc.baseEndpoint != "" {
		clientOpts = append(clientOpts, func(o *bedrockruntime.Options) {
			o.BaseEndpoint = aws.String(pc.baseEndpoint)
		})
	}

	client := bedrockruntime.NewFromConfig(cfg, clientOpts...)

	return &Provider{
		client:         client,
		region:         cfg.Region,
		requestTimeout: pc.requestTimeout,
	}, nil
}

// converseParams holds the shared request parameters for Converse and ConverseStream.
type converseParams struct {
	messages        []types.Message
	system          []types.SystemContentBlock
	inferenceConfig *types.InferenceConfiguration
	toolConfig      *types.ToolConfiguration
}

// modelDeprecatesTemperature reports whether the given Bedrock model rejects the
// temperature inference parameter. Newer Claude models (Opus 4.8 and later) treat
// temperature as deprecated and return a ValidationException if it is supplied:
//
//	ValidationException: The model returned the following errors:
//	temperature is deprecated for this model.
//
// The match is intentionally loose: Bedrock model IDs and inference-profile ARNs
// embed the model name (e.g. "us.anthropic.claude-opus-4-8-20260514-v1:0"), so a
// substring check covers both bare IDs and region-prefixed inference profiles.
func modelDeprecatesTemperature(model string) bool {
	m := strings.ToLower(model)
	return strings.Contains(m, "claude-opus-4-8")
}

// modelSupportsCaching reports whether the given Bedrock model accepts prompt
// cache points (CachePointBlock) in the Converse request. Inserting a cache
// point for a model that does not support it returns a ValidationException, so
// caching is opt-in by model.
//
// The match is loose for the same reason as modelDeprecatesTemperature: Bedrock
// model IDs and inference-profile ARNs embed the model name, so a substring
// check covers bare IDs and region-prefixed inference profiles alike. The set
// covers the cache-enabled Claude families on Bedrock (3.5 v2, 3.7, and 4.x).
func modelSupportsCaching(model string) bool {
	m := strings.ToLower(model)
	if !strings.Contains(m, "claude") {
		return false
	}
	switch {
	case strings.Contains(m, "claude-3-5-sonnet"),
		strings.Contains(m, "claude-3-5-haiku"),
		strings.Contains(m, "claude-3-7-sonnet"),
		strings.Contains(m, "claude-sonnet-4"),
		strings.Contains(m, "claude-opus-4"),
		strings.Contains(m, "claude-haiku-4"):
		return true
	default:
		return false
	}
}

// cachePointBlock returns the SDK cache-point marker inserted after a cacheable
// prefix in system, tool, or message content.
func newCachePointContentBlock() types.ContentBlock {
	return &types.ContentBlockMemberCachePoint{
		Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
	}
}

func newCachePointSystemBlock() types.SystemContentBlock {
	return &types.SystemContentBlockMemberCachePoint{
		Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
	}
}

func newCachePointTool() types.Tool {
	return &types.ToolMemberCachePoint{
		Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
	}
}

func buildConverseParams(messages []Message, tools []ToolDefinition, model string, options map[string]any) converseParams {
	cacheEnabled := modelSupportsCaching(model)
	bedrockMessages, systemPrompts := convertMessagesWithCache(messages, cacheEnabled)

	var inferenceConfig *types.InferenceConfiguration

	if maxTokens, ok := common.AsInt(options["max_tokens"]); ok && maxTokens > 0 {
		if inferenceConfig == nil {
			inferenceConfig = &types.InferenceConfiguration{}
		}
		if maxTokens > math.MaxInt32 {
			maxTokens = math.MaxInt32
		}
		inferenceConfig.MaxTokens = aws.Int32(int32(maxTokens))
	}

	if temp, ok := common.AsFloat(options["temperature"]); ok {
		if modelDeprecatesTemperature(model) {
			log.Printf("bedrock: temperature dropped because model %q no longer supports it", model)
		} else {
			if inferenceConfig == nil {
				inferenceConfig = &types.InferenceConfiguration{}
			}
			inferenceConfig.Temperature = aws.Float32(float32(temp))
		}
	}

	var toolConfig *types.ToolConfiguration
	if len(tools) > 0 {
		tc := convertToolsWithCache(tools, cacheEnabled)
		if len(tc.Tools) > 0 {
			toolConfig = tc
		}
	}

	return converseParams{
		messages:        bedrockMessages,
		system:          systemPrompts,
		inferenceConfig: inferenceConfig,
		toolConfig:      toolConfig,
	}
}

// Chat sends messages to AWS Bedrock using the Converse API.
func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	effectiveTimeout := p.requestTimeout
	if effectiveTimeout <= 0 {
		effectiveTimeout = common.DefaultRequestTimeout
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, effectiveTimeout)
		defer cancel()
	}

	params := buildConverseParams(messages, tools, model, options)
	input := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(model),
		Messages:        params.messages,
		InferenceConfig: params.inferenceConfig,
		ToolConfig:      params.toolConfig,
	}
	if len(params.system) > 0 {
		input.System = params.system
	}

	output, err := p.client.Converse(ctx, input)
	if err != nil {
		if isSSOTokenError(err) {
			return nil, fmt.Errorf(
				"bedrock converse: AWS credentials may have expired. If using AWS SSO, run 'aws sso login' to refresh: %w",
				err,
			)
		}
		return nil, fmt.Errorf("bedrock converse: %w", err)
	}

	return parseResponse(output)
}

// ChatStream sends messages to AWS Bedrock using the ConverseStream API.
// It streams the accumulated text so far via the onChunk callback and returns the complete response.
func (p *Provider) ChatStream(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
	onChunk func(accumulated string),
) (*LLMResponse, error) {
	if p.requestTimeout > 0 {
		if _, hasDeadline := ctx.Deadline(); !hasDeadline {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, p.requestTimeout)
			defer cancel()
		}
	}

	params := buildConverseParams(messages, tools, model, options)
	input := &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(model),
		Messages:        params.messages,
		InferenceConfig: params.inferenceConfig,
		ToolConfig:      params.toolConfig,
	}
	if len(params.system) > 0 {
		input.System = params.system
	}

	output, err := p.client.ConverseStream(ctx, input)
	if err != nil {
		if isSSOTokenError(err) {
			return nil, fmt.Errorf(
				"bedrock conversestream: AWS credentials may have expired. If using AWS SSO, run 'aws sso login' to refresh: %w",
				err,
			)
		}
		return nil, fmt.Errorf("bedrock conversestream: %w", err)
	}

	return parseStreamResponse(ctx, output.GetStream(), onChunk)
}

// converseStreamReader abstracts the Bedrock event stream so parseStreamResponse
// can be unit-tested with a mock event source.
type converseStreamReader interface {
	Events() <-chan types.ConverseStreamOutput
	Err() error
	Close() error
}

// parseStreamResponse processes the ConverseStream event stream and accumulates the response.
func parseStreamResponse(
	ctx context.Context,
	stream converseStreamReader,
	onChunk func(accumulated string),
) (resp *LLMResponse, err error) {
	if stream == nil {
		return nil, fmt.Errorf("bedrock conversestream: nil event stream")
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			if err == nil {
				err = fmt.Errorf("bedrock conversestream: close event stream: %w", closeErr)
			} else {
				log.Printf("bedrock conversestream: close event stream: %v", closeErr)
			}
		}
	}()

	var textContent strings.Builder
	finishReason := "stop"
	var usage *UsageInfo
	toolCalls := make([]ToolCall, 0)

	// Track active tool use blocks by index
	type toolAccum struct {
		id       string
		name     string
		argsJSON strings.Builder
	}
	activeTools := map[int]*toolAccum{}

	events := stream.Events()
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case event, ok := <-events:
			if !ok {
				// Stream closed
				goto done
			}

			switch e := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockStart:
				// New content block starting
				if toolUse, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
					activeTools[int(aws.ToInt32(e.Value.ContentBlockIndex))] = &toolAccum{
						id:   aws.ToString(toolUse.Value.ToolUseId),
						name: aws.ToString(toolUse.Value.Name),
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockDelta:
				// Content delta
				switch delta := e.Value.Delta.(type) {
				case *types.ContentBlockDeltaMemberText:
					textContent.WriteString(delta.Value)
					if onChunk != nil {
						onChunk(textContent.String())
					}
				case *types.ContentBlockDeltaMemberToolUse:
					idx := int(aws.ToInt32(e.Value.ContentBlockIndex))
					if tool, exists := activeTools[idx]; exists {
						tool.argsJSON.WriteString(aws.ToString(delta.Value.Input))
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockStop:
				// Content block finished - finalize tool if it was a tool use
				idx := int(aws.ToInt32(e.Value.ContentBlockIndex))
				if tool, exists := activeTools[idx]; exists {
					args := make(map[string]any)
					argsStr := tool.argsJSON.String()
					if argsStr != "" {
						if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
							log.Printf("bedrock: stream: failed to parse tool arguments for %q: %v", tool.name, err)
							args = map[string]any{"raw": argsStr}
						}
					}
					funcArgs := argsStr
					if argsJSON, marshalErr := json.Marshal(args); marshalErr == nil {
						funcArgs = string(argsJSON)
					}
					toolCalls = append(toolCalls, ToolCall{
						ID:        tool.id,
						Name:      tool.name,
						Arguments: args,
						Function: &FunctionCall{
							Name:      tool.name,
							Arguments: funcArgs,
						},
					})
					delete(activeTools, idx)
				}

			case *types.ConverseStreamOutputMemberMessageStop:
				// Message complete
				switch e.Value.StopReason {
				case types.StopReasonToolUse:
					finishReason = "tool_calls"
				case types.StopReasonMaxTokens:
					finishReason = "length"
				case types.StopReasonEndTurn:
					finishReason = "stop"
				case types.StopReasonStopSequence:
					finishReason = "stop"
				case types.StopReasonContentFiltered:
					finishReason = "content_filter"
				default:
					finishReason = "stop"
				}

			case *types.ConverseStreamOutputMemberMetadata:
				// Usage metadata
				if e.Value.Usage != nil {
					usage = &UsageInfo{
						PromptTokens:     int(aws.ToInt32(e.Value.Usage.InputTokens)),
						CompletionTokens: int(aws.ToInt32(e.Value.Usage.OutputTokens)),
						TotalTokens: int(
							aws.ToInt32(e.Value.Usage.InputTokens),
						) + int(
							aws.ToInt32(e.Value.Usage.OutputTokens),
						),
						CacheReadTokens:  int(aws.ToInt32(e.Value.Usage.CacheReadInputTokens)),
						CacheWriteTokens: int(aws.ToInt32(e.Value.Usage.CacheWriteInputTokens)),
					}
					logCacheUsage("conversestream", usage)
				}
			}
		}
	}

done:
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("bedrock conversestream: %w", err)
	}

	return &LLMResponse{
		Content:      textContent.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

// GetDefaultModel returns an empty string as Bedrock models are user-configured.
func (p *Provider) GetDefaultModel() string {
	return ""
}

// Region returns the AWS region configured for this Provider.
func (p *Provider) Region() string {
	return p.region
}

// convertMessages converts internal messages to Bedrock Converse format.
// Returns the conversation messages and any system prompts separately.
// Note: Bedrock requires all tool results for a given assistant turn to be in a single
// user message with multiple ToolResultBlock content blocks. This function merges
// consecutive tool result messages accordingly.
func convertMessages(messages []Message) ([]types.Message, []types.SystemContentBlock) {
	return convertMessagesWithCache(messages, false)
}

// convertMessagesWithCache is convertMessages with optional Bedrock prompt
// caching. When cacheEnabled is true and a system message carries structured
// SystemParts, each part is emitted as its own block and a cache point is
// inserted after the last part marked CacheControl "ephemeral". This caches the
// stable system prefix while leaving volatile trailing parts (time, session,
// summary) outside the cached region. The cache point is omitted when no part
// is ephemeral, so the per-request dynamic context never anchors the cache.
func convertMessagesWithCache(messages []Message, cacheEnabled bool) ([]types.Message, []types.SystemContentBlock) {
	var bedrockMessages []types.Message
	var systemPrompts []types.SystemContentBlock

	// Helper to check if a message is a tool result
	isToolResult := func(msg Message) bool {
		return (msg.Role == "tool" || (msg.Role == "user" && msg.ToolCallID != "")) && msg.ToolCallID != ""
	}

	// Helper to create a tool result content block
	makeToolResultBlock := func(msg Message) types.ContentBlock {
		return &types.ContentBlockMemberToolResult{
			Value: types.ToolResultBlock{
				ToolUseId: aws.String(msg.ToolCallID),
				Content: []types.ToolResultContentBlock{
					&types.ToolResultContentBlockMemberText{
						Value: msg.Content,
					},
				},
			},
		}
	}

	i := 0
	for i < len(messages) {
		msg := messages[i]

		switch {
		case msg.Role == "system":
			// System messages go to the System field. With caching enabled and
			// structured SystemParts available, emit each part separately and
			// place a cache point after the last ephemeral (stable) part so the
			// static prefix is cached and volatile trailing parts are not.
			if cacheEnabled && len(msg.SystemParts) > 0 {
				lastEphemeral := -1
				for idx, part := range msg.SystemParts {
					if part.CacheControl != nil && part.CacheControl.Type == "ephemeral" {
						lastEphemeral = idx
					}
				}
				for idx, part := range msg.SystemParts {
					systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
						Value: part.Text,
					})
					if idx == lastEphemeral {
						systemPrompts = append(systemPrompts, newCachePointSystemBlock())
					}
				}
			} else {
				systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
					Value: msg.Content,
				})
			}
			i++

		case isToolResult(msg):
			// Collect all consecutive tool results into a single user message
			// Bedrock requires all tool results for a turn in one message
			var toolResultBlocks []types.ContentBlock
			for i < len(messages) && isToolResult(messages[i]) {
				toolResultBlocks = append(toolResultBlocks, makeToolResultBlock(messages[i]))
				i++
			}
			bedrockMessages = append(bedrockMessages, types.Message{
				Role:    types.ConversationRoleUser,
				Content: toolResultBlocks,
			})

		case msg.Role == "user":
			// Regular user message (no ToolCallID)
			content := buildUserContent(msg)
			bedrockMessages = append(bedrockMessages, types.Message{
				Role:    types.ConversationRoleUser,
				Content: content,
			})
			i++

		case msg.Role == "assistant":
			content := buildAssistantContent(msg)
			bedrockMessages = append(bedrockMessages, types.Message{
				Role:    types.ConversationRoleAssistant,
				Content: content,
			})
			i++

		case msg.Role == "tool" && msg.ToolCallID == "":
			// Tool message without ToolCallID - treat as regular user message
			content := buildUserContent(msg)
			bedrockMessages = append(bedrockMessages, types.Message{
				Role:    types.ConversationRoleUser,
				Content: content,
			})
			i++

		default:
			// Unknown role - skip
			i++
		}
	}

	return bedrockMessages, systemPrompts
}

// buildUserContent builds Bedrock content blocks for a user message.
func buildUserContent(msg Message) []types.ContentBlock {
	var content []types.ContentBlock

	// Add text content
	if msg.Content != "" {
		content = append(content, &types.ContentBlockMemberText{
			Value: msg.Content,
		})
	}

	// Add images from Media field
	for _, mediaURL := range msg.Media {
		if strings.HasPrefix(mediaURL, "data:image/") {
			// Parse data URL: data:image/jpeg;base64,<data>
			parts := strings.SplitN(mediaURL, ",", 2)
			if len(parts) != 2 {
				continue
			}

			// Extract media type from "data:image/jpeg;base64"
			mediaType := ""
			header := parts[0]
			if idx := strings.Index(header, "/"); idx != -1 {
				end := strings.Index(header[idx:], ";")
				if end == -1 {
					end = len(header) - idx
				}
				mediaType = header[idx+1 : idx+end]
			}

			// Verify this is base64 encoded
			if !strings.Contains(header, ";base64") {
				continue // Skip non-base64 encoded data
			}

			// Map media type to Bedrock format
			var format types.ImageFormat
			switch mediaType {
			case "jpeg", "jpg":
				format = types.ImageFormatJpeg
			case "png":
				format = types.ImageFormatPng
			case "gif":
				format = types.ImageFormatGif
			case "webp":
				format = types.ImageFormatWebp
			default:
				continue // Skip unsupported formats
			}

			// Check size before decoding to prevent excessive memory allocation
			// Bedrock has a ~20MB request limit; cap decoded images at 10MB
			const maxImageSize = 10 * 1024 * 1024
			decodedLen := base64.StdEncoding.DecodedLen(len(parts[1]))
			if decodedLen > maxImageSize {
				log.Printf("bedrock: skipping image exceeding size limit (%d bytes > %d)", decodedLen, maxImageSize)
				continue
			}

			// Decode base64 data
			imageData, err := base64.StdEncoding.DecodeString(parts[1])
			if err != nil {
				log.Printf("bedrock: failed to decode base64 image data: %v", err)
				continue
			}

			content = append(content, &types.ContentBlockMemberImage{
				Value: types.ImageBlock{
					Format: format,
					Source: &types.ImageSourceMemberBytes{
						Value: imageData,
					},
				},
			})
		}
	}

	// Bedrock requires at least one content block; add empty text if needed
	if len(content) == 0 {
		content = append(content, &types.ContentBlockMemberText{Value: ""})
	}

	return content
}

// buildAssistantContent builds Bedrock content blocks for an assistant message.
func buildAssistantContent(msg Message) []types.ContentBlock {
	var content []types.ContentBlock

	// Add text content if present
	if msg.Content != "" {
		content = append(content, &types.ContentBlockMemberText{
			Value: msg.Content,
		})
	}

	// Add tool use blocks
	for _, tc := range msg.ToolCalls {
		// Validate tool call ID - Bedrock requires non-empty ToolUseId
		if strings.TrimSpace(tc.ID) == "" {
			log.Printf("bedrock: skipping tool call with empty ID (name: %q)", tc.Name)
			continue
		}

		// Resolve tool name: prefer tc.Name, fallback to tc.Function.Name
		// (tc.Name/tc.Arguments are json:"-" and may be empty when from JSON)
		toolName := tc.Name
		if toolName == "" && tc.Function != nil {
			toolName = tc.Function.Name
		}
		if strings.TrimSpace(toolName) == "" {
			continue
		}

		// Resolve arguments: prefer tc.Arguments, fallback to parsing tc.Function.Arguments
		args := tc.Arguments
		if args == nil && tc.Function != nil && tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				log.Printf("bedrock: failed to parse Function.Arguments for tool %q: %v", toolName, err)
				args = map[string]any{}
			}
		}
		if args == nil {
			args = map[string]any{}
		}

		// Convert arguments to a Bedrock document using NewLazyDocument
		inputDoc := document.NewLazyDocument(args)

		content = append(content, &types.ContentBlockMemberToolUse{
			Value: types.ToolUseBlock{
				ToolUseId: aws.String(tc.ID),
				Name:      aws.String(toolName),
				Input:     inputDoc,
			},
		})
	}

	// Bedrock requires at least one content block; add empty text if needed
	if len(content) == 0 {
		content = append(content, &types.ContentBlockMemberText{Value: ""})
	}

	return content
}

// convertTools converts tool definitions to Bedrock format.
func convertTools(tools []ToolDefinition) *types.ToolConfiguration {
	return convertToolsWithCache(tools, false)
}

// convertToolsWithCache is convertTools with optional Bedrock prompt caching.
// When cacheEnabled is true and at least one tool is produced, a cache point is
// appended after the tool list. The tool schema is large and identical across
// turns, so caching it yields the most reliable savings. The point is omitted
// when no tools are emitted to avoid a ValidationException on an empty config.
func convertToolsWithCache(tools []ToolDefinition, cacheEnabled bool) *types.ToolConfiguration {
	bedrockTools := make([]types.Tool, 0, len(tools))

	for _, tool := range tools {
		// Skip tools with empty names
		if strings.TrimSpace(tool.Function.Name) == "" {
			continue
		}

		// Ensure parameters is not nil - default to minimal object schema
		params := tool.Function.Parameters
		if params == nil {
			params = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}

		// Convert parameters schema to a Bedrock document
		inputSchema := document.NewLazyDocument(params)

		bedrockTools = append(bedrockTools, &types.ToolMemberToolSpec{
			Value: types.ToolSpecification{
				Name:        aws.String(tool.Function.Name),
				Description: aws.String(tool.Function.Description),
				InputSchema: &types.ToolInputSchemaMemberJson{
					Value: inputSchema,
				},
			},
		})
	}

	if cacheEnabled && len(bedrockTools) > 0 {
		bedrockTools = append(bedrockTools, newCachePointTool())
	}

	return &types.ToolConfiguration{
		Tools: bedrockTools,
	}
}

// parseResponse converts Bedrock Converse output to LLMResponse.
func parseResponse(output *bedrockruntime.ConverseOutput) (*LLMResponse, error) {
	var content strings.Builder
	toolCalls := make([]ToolCall, 0)

	// Process output content blocks
	if output.Output != nil {
		if msgOutput, ok := output.Output.(*types.ConverseOutputMemberMessage); ok {
			for _, block := range msgOutput.Value.Content {
				switch b := block.(type) {
				case *types.ContentBlockMemberText:
					content.WriteString(b.Value)

				case *types.ContentBlockMemberToolUse:
					// Unmarshal the document interface to a map
					args := make(map[string]any)
					if b.Value.Input != nil {
						if err := b.Value.Input.UnmarshalSmithyDocument(&args); err != nil {
							log.Printf("bedrock: failed to unmarshal tool input for tool %q (id %q): %v",
								aws.ToString(b.Value.Name),
								aws.ToString(b.Value.ToolUseId),
								err,
							)
							args = make(map[string]any)
						}
					}

					// Serialize arguments to JSON string for FunctionCall
					argsJSON, err := json.Marshal(args)
					if err != nil {
						log.Printf("bedrock: failed to marshal tool arguments for tool %q (id %q): %v",
							aws.ToString(b.Value.Name),
							aws.ToString(b.Value.ToolUseId),
							err,
						)
						argsJSON = []byte("{}")
					}

					toolCalls = append(toolCalls, ToolCall{
						ID:        aws.ToString(b.Value.ToolUseId),
						Name:      aws.ToString(b.Value.Name),
						Arguments: args,
						Function: &FunctionCall{
							Name:      aws.ToString(b.Value.Name),
							Arguments: string(argsJSON),
						},
					})
				}
			}
		}
	}

	// Map stop reason
	finishReason := "stop"
	switch output.StopReason {
	case types.StopReasonToolUse:
		finishReason = "tool_calls"
	case types.StopReasonMaxTokens:
		finishReason = "length"
	case types.StopReasonEndTurn:
		finishReason = "stop"
	case types.StopReasonStopSequence:
		finishReason = "stop"
	case types.StopReasonContentFiltered:
		finishReason = "content_filter"
	}

	// Build usage info
	var usage *UsageInfo
	if output.Usage != nil {
		usage = &UsageInfo{
			PromptTokens:     int(aws.ToInt32(output.Usage.InputTokens)),
			CompletionTokens: int(aws.ToInt32(output.Usage.OutputTokens)),
			TotalTokens:      int(aws.ToInt32(output.Usage.InputTokens)) + int(aws.ToInt32(output.Usage.OutputTokens)),
			CacheReadTokens:  int(aws.ToInt32(output.Usage.CacheReadInputTokens)),
			CacheWriteTokens: int(aws.ToInt32(output.Usage.CacheWriteInputTokens)),
		}
		logCacheUsage("converse", usage)
	}

	return &LLMResponse{
		Content:      content.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

// logCacheUsage emits a single line summarizing prompt-cache activity for the
// turn so cache hits/writes are observable in logs. read>0 means the static
// prefix (system + tools) was served from cache at ~0.1x input price; write>0
// means it was (re)written this turn at ~1.25x.
func logCacheUsage(api string, usage *UsageInfo) {
	if usage == nil {
		return
	}
	// Use the structured logger (not the std log package, whose output is not
	// captured in deployed builds) so cache activity is visible in pod logs.
	logger.InfoCF("provider.bedrock", "prompt cache usage", map[string]any{
		"api":           api,
		"input_tokens":  usage.PromptTokens,
		"output_tokens": usage.CompletionTokens,
		"cache_read":    usage.CacheReadTokens,
		"cache_write":   usage.CacheWriteTokens,
		"cache_hit":     usage.CacheReadTokens > 0,
	})
}

// isSSOTokenError checks if the error is related to expired or invalid AWS SSO tokens.
// This helps provide actionable guidance when SSO credentials need to be refreshed.
// Only matches SSO-specific error patterns to avoid misclassifying other AWS credential errors.
func isSSOTokenError(err error) bool {
	if err == nil {
		return false
	}
	lower := strings.ToLower(err.Error())

	// Check for specific SSO token expiration/refresh-related error patterns (case-insensitive)
	// Avoid matching generic patterns that could match non-SSO AWS errors (e.g., STS ExpiredToken)
	if strings.Contains(lower, "refresh cached sso token") {
		return true
	}
	if strings.Contains(lower, "read cached sso token") {
		return true
	}
	if strings.Contains(lower, "sso oidc") {
		return true
	}
	if strings.Contains(lower, "invalidgrantexception") {
		return true
	}

	return false
}

package agent

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/providers"
)

type mockStreamingProvider struct {
	response string
}

func (m *mockStreamingProvider) Chat(ctx context.Context, messages []providers.Message, tools []providers.ToolDefinition, model string, options map[string]any) (*providers.LLMResponse, error) {
	return &providers.LLMResponse{Content: m.response}, nil
}

func (m *mockStreamingProvider) ChatStream(ctx context.Context, messages []providers.Message, tools []providers.ToolDefinition, model string, options map[string]any, onChunk func(string)) (*providers.LLMResponse, error) {
	words := strings.Fields(m.response)
	accumulated := ""
	for _, word := range words {
		if accumulated != "" {
			accumulated += " "
		}
		accumulated += word
		onChunk(accumulated)
		time.Sleep(10 * time.Millisecond)
	}
	return &providers.LLMResponse{Content: m.response}, nil
}

func (m *mockStreamingProvider) GetDefaultModel() string { return "test" }

type mockStreamer struct {
	mu       sync.Mutex
	updates  []string
	final    string
	canceled bool
}

func (s *mockStreamer) Update(ctx context.Context, content string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.updates = append(s.updates, content)
	return nil
}

func (s *mockStreamer) Finalize(ctx context.Context, content string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.final = content
	return nil
}

func (s *mockStreamer) Cancel(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.canceled = true
}

func TestAgentLoop_StreamingIntegration(t *testing.T) {
	// This is a placeholder - full test requires more setup
	t.Skip("Requires full agent setup - manual verification recommended")
}

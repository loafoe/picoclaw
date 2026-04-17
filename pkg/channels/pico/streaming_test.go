package pico

import (
	"context"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/channels"
	"github.com/sipeed/picoclaw/pkg/config"
)

func TestPicoChannel_ImplementsStreamingCapable(t *testing.T) {
	bc := &config.Channel{Enabled: true}
	cfg := &config.PicoSettings{Token: *config.NewSecureString("test-token")}
	msgBus := bus.NewMessageBus()

	ch, err := NewPicoChannel(bc, cfg, msgBus)
	if err != nil {
		t.Fatalf("NewPicoChannel() error = %v", err)
	}

	var _ channels.StreamingCapable = ch
}

func TestPicoStreamer_UpdateAndFinalize(t *testing.T) {
	bc := &config.Channel{Enabled: true}
	cfg := &config.PicoSettings{Token: *config.NewSecureString("test-token")}
	msgBus := bus.NewMessageBus()

	ch, err := NewPicoChannel(bc, cfg, msgBus)
	if err != nil {
		t.Fatalf("NewPicoChannel() error = %v", err)
	}
	ch.SetRunning(true)

	ctx := context.Background()
	chatID := "pico:test-session"

	// Test the streamer methods don't panic even without active connections
	s := &picoStreamer{
		channel:   ch,
		chatID:    chatID,
		messageID: "test-msg-id",
	}

	// Update should not panic even without connections (returns error from broadcast)
	_ = s.Update(ctx, "partial content")
	_ = s.Update(ctx, "more content")
	_ = s.Finalize(ctx, "final content")

	// After finalize, updates should be no-op (return nil)
	if err := s.Update(ctx, "ignored"); err != nil {
		t.Errorf("Update after Finalize should not error, got %v", err)
	}
}

func TestPicoStreamer_Cancel(t *testing.T) {
	s := &picoStreamer{
		messageID: "test-msg",
	}

	ctx := context.Background()
	s.Cancel(ctx)

	if !s.finalized {
		t.Error("Cancel should set finalized to true")
	}

	// After cancel, Update should be no-op (return nil)
	if err := s.Update(ctx, "ignored"); err != nil {
		t.Errorf("Update after Cancel should not error, got %v", err)
	}
}

package pico

import (
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

package grafana_alertmanager

import (
	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/channels"
	"github.com/sipeed/picoclaw/pkg/config"
)

func init() {
	channels.RegisterFactory(
		config.ChannelGrafanaAlertmanager,
		func(channelName, channelType string, cfg *config.Config, b *bus.MessageBus) (channels.Channel, error) {
			bc := cfg.Channels[channelName]
			decoded, err := bc.GetDecoded()
			if err != nil {
				return nil, err
			}
			c, ok := decoded.(*config.GrafanaAlertmanagerConfig)
			if !ok {
				return nil, channels.ErrSendFailed
			}
			ch, err := NewGrafanaAlertmanagerChannel(bc, c, b)
			if err != nil {
				return nil, err
			}
			if channelName != config.ChannelGrafanaAlertmanager {
				ch.SetName(channelName)
			}
			return ch, nil
		},
	)
}

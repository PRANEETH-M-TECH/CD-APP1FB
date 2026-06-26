import React from 'react';
import { Connection, Asset } from './types';

const THEME_COLORS = {
  indigo: '#818cf8',
  gold: '#fbbf24',
  emerald: '#34d399',
  rose: '#fb7185',
};

interface ConnectionsViewProps {
  connections: Connection[];
  assets: Asset[];
  theme: 'indigo' | 'gold' | 'emerald' | 'rose';
}

export const ConnectionsView: React.FC<ConnectionsViewProps> = ({
  connections = [],
  assets = [],
  theme,
}) => {
  const color = THEME_COLORS[theme] || THEME_COLORS.indigo;

  // Map assets by ID for quick center coordinate lookups
  const assetMap = React.useMemo(() => {
    const map = new Map<string, Asset>();
    assets.forEach((asset) => {
      map.set(asset.id, asset);
    });
    return map;
  }, [assets]);

  // Filter connections where both source and destination exist in current scene
  const activeConnections = React.useMemo(() => {
    return connections.filter((conn) => {
      return assetMap.has(conn.from) && assetMap.has(conn.to);
    });
  }, [connections, assetMap]);

  if (activeConnections.length === 0) {
    return null;
  }

  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 5,
      }}
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      <defs>
        <marker
          id={`arrowhead-${theme}`}
          viewBox="0 0 10 10"
          refX="6"
          refY="5"
          markerWidth="4"
          markerHeight="4"
          orient="auto-start-reverse"
        >
          <path d="M 0 2 L 8 5 L 0 8 z" fill={color} />
        </marker>
      </defs>

      {activeConnections.map((conn, idx) => {
        const fromAsset = assetMap.get(conn.from)!;
        const toAsset = assetMap.get(conn.to)!;

        const x1 = fromAsset.layout.left + fromAsset.layout.width / 2;
        const y1 = fromAsset.layout.top + fromAsset.layout.height / 2;

        const x2 = toAsset.layout.left + toAsset.layout.width / 2;
        const y2 = toAsset.layout.top + toAsset.layout.height / 2;

        const isArrow = conn.type === 'arrow';

        return (
          <line
            key={`conn-${conn.from}-${conn.to}-${idx}`}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={color}
            strokeWidth="0.6"
            strokeDasharray={conn.type === 'line' ? '1,1' : undefined}
            markerEnd={isArrow ? `url(#arrowhead-${theme})` : undefined}
            style={{
              opacity: 0.6,
            }}
          />
        );
      })}
    </svg>
  );
};

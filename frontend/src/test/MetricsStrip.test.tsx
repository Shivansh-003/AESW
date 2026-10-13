import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricsStrip } from '../components/MetricsStrip';
import { MetricsDto } from '../lib/api';

describe('MetricsStrip Component', () => {
  const baseMetrics: MetricsDto = {
    algorithm: 'aesw',
    search_time: 10,
    status: 'RUNNING',
    success: false,
    total_moves: 25,
    node_revisits: 5,
    nodes_visited: 18,
    total_messages: 12,
    total_cost: 26.2,
    c_m: 0.1,
  };

  it('renders waiting message when metrics is null', () => {
    render(<MetricsStrip metrics={null} totalNodes={30} />);
    expect(screen.getByText(/Waiting for simulation initialization/i)).toBeInTheDocument();
  });

  it('renders AESW metrics without No comm badge or baseline tooltip', () => {
    render(
      <MetricsStrip
        metrics={baseMetrics}
        totalNodes={30}
        isAesw={true}
        algorithm="aesw"
      />
    );

    const messagesLabel = screen.getByText(/Messages \(Q\):/i);
    expect(messagesLabel).toBeInTheDocument();
    expect(messagesLabel.parentElement).toHaveTextContent('12');
    expect(screen.queryByText('No comm')).not.toBeInTheDocument();
  });

  it('renders communication-free baseline (Random Walk) with count and explanatory No comm badge and tooltip', () => {
    const rwMetrics: MetricsDto = {
      ...baseMetrics,
      algorithm: 'random_walk',
      total_messages: 0,
      total_cost: 25.0,
    };

    render(
      <MetricsStrip
        metrics={rwMetrics}
        totalNodes={30}
        isAesw={false}
        algorithm="random_walk"
      />
    );

    const messagesLabel = screen.getByText(/Messages \(Q\):/i);
    expect(messagesLabel).toBeInTheDocument();
    expect(messagesLabel.parentElement).toHaveTextContent('0');

    const badge = screen.getByText('No comm');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveAttribute(
      'title',
      'Communication-free baseline — this algorithm does not exchange messages (Q = 0 by design).'
    );
    expect(messagesLabel.parentElement).toHaveAttribute(
      'title',
      'Communication-free baseline — this algorithm does not exchange messages (Q = 0 by design).'
    );
  });

  it('renders communication-free baseline (Degree-based Walk) correctly based on algorithm metadata', () => {
    const degreeMetrics: MetricsDto = {
      ...baseMetrics,
      algorithm: 'degree',
      total_messages: 0,
      total_cost: 20.0,
    };

    render(
      <MetricsStrip
        metrics={degreeMetrics}
        totalNodes={30}
        isAesw={false}
        algorithm="degree"
      />
    );

    const messagesLabel = screen.getByText(/Messages \(Q\):/i);
    expect(messagesLabel.parentElement).toHaveTextContent('0');
    expect(screen.getByText('No comm')).toBeInTheDocument();
  });

  it('preserves cost formula rendering (M + c_m * Q)', () => {
    render(
      <MetricsStrip
        metrics={baseMetrics}
        totalNodes={30}
        isAesw={true}
        algorithm="aesw"
      />
    );

    expect(screen.getByText(/Total Cost \(C\):/i)).toBeInTheDocument();
    expect(screen.getByText('26.20')).toBeInTheDocument();
    expect(screen.getByText(/\(M \+ 0\.1 × Q\)/i)).toBeInTheDocument();
  });
});

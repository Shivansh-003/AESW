import '@testing-library/jest-dom';

// Polyfill window.fetch or ResizeObserver for testing environment if needed
if (!globalThis.fetch) {
  // @ts-ignore
  globalThis.fetch = vi.fn();
}

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverMock;

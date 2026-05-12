/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import App from '../App';

jest.mock('react-native-vision-camera', () => {
  const MockReact = require('react');
  const { View } = require('react-native');

  return {
    Camera: MockReact.forwardRef((props: object, ref: React.Ref<unknown>) => (
      <View {...props} ref={ref} />
    )),
    useCameraDevice: jest.fn(() => ({ id: 'back-camera' })),
    useCameraPermission: jest.fn(() => ({
      hasPermission: true,
      requestPermission: jest.fn(async () => true),
    })),
    usePhotoOutput: jest.fn(() => ({
      capturePhotoToFile: jest.fn(),
    })),
  };
});

beforeEach(() => {
  globalThis.fetch = jest.fn(async () => ({
    ok: true,
    text: jest.fn(async () => JSON.stringify({ status: 'ok' })),
  })) as unknown as typeof fetch;
});

test('renders correctly', async () => {
  let renderer: ReactTestRenderer.ReactTestRenderer | null = null;

  await ReactTestRenderer.act(async () => {
    renderer = ReactTestRenderer.create(<App />);
    await Promise.resolve();
    await Promise.resolve();
  });

  await ReactTestRenderer.act(async () => {
    renderer?.unmount();
  });
});

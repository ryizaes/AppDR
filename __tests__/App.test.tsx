/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import App, { CLASS_LABELS, formatClassValue } from '../App';

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

test('shows capture and fundus upload actions on the main screen', async () => {
  let renderer!: ReactTestRenderer.ReactTestRenderer;

  await ReactTestRenderer.act(async () => {
    renderer = ReactTestRenderer.create(<App />);
    await Promise.resolve();
    await Promise.resolve();
  });

  const rendered = JSON.stringify(renderer.toJSON());
  expect(rendered).toContain('Capture Image');
  expect(rendered).toContain('Upload Fundus Image');

  await ReactTestRenderer.act(async () => {
    renderer.unmount();
  });
});

test('uses medical severity labels without numeric class wording', () => {
  expect(CLASS_LABELS).toEqual({
    0: 'No apparent diabetic retinopathy',
    1: 'Mild non-proliferative diabetic retinopathy',
    2: 'Moderate non-proliferative diabetic retinopathy',
    3: 'Severe non-proliferative diabetic retinopathy',
    4: 'Proliferative diabetic retinopathy',
  });

  Object.values(CLASS_LABELS).forEach(label => {
    expect(label).not.toMatch(/^(Class|Grade)\s+[0-4]$/i);
  });
  expect(formatClassValue(99)).toBe('Medical severity label unavailable');
});

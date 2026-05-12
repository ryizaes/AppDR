# Tests

This folder contains the React Native smoke test.

## App Test

`App.test.tsx` renders the root app component to catch basic runtime errors.
Native camera code is mocked because Jest cannot load the real Android/iOS
camera module.

The test also mocks `fetch` so the automatic backend health check can complete
without requiring FastAPI to run during unit tests.

## Run

From the project root:

```powershell
npm.cmd test -- --runInBand
```

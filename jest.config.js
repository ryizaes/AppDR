module.exports = {
  preset: '@react-native/jest-preset',
  modulePathIgnorePatterns: [
    '<rootDir>/backend/.venv',
    '<rootDir>/backend/backups',
  ],
  testTimeout: 15000,
};

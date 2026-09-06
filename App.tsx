import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  NativeModules,
  PermissionsAndroid,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import {
  Camera,
  CameraRef,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';

type AppScreen = 'home' | 'capture' | 'result' | 'history' | 'about' | 'tips';

type CapturedImage = {
  id: string;
  path: string;
  uri: string;
  analysisPath: string;
  analysisUri: string;
  fileName: string;
  mimeType: string;
  savedUri: string | null;
  savedAt: string;
  source: 'capture' | 'upload';
  analysis?: AnalyzeResponse;
};

type GallerySaverModule = {
  saveImage: (filePath: string, albumName: string) => Promise<string>;
};

type ImageCropModule = {
  cropCenterSquare: (filePath: string, cropScale: number) => Promise<string>;
};

type PickedImage = {
  filePath?: string;
  fileUri?: string;
  name?: string;
  type?: string;
};

type ImagePickerModule = {
  pickImage: () => Promise<PickedImage>;
};

type SourceCodeModule = {
  scriptURL?: string;
};

type BackendConnectionState = 'idle' | 'checking' | 'connected' | 'offline';

type BackendModelStatus = {
  model_mode?: string;
  dual_model_ready?: boolean;
  demo_hybrid_ready?: boolean;
  demo_models_dir?: string;
  rollback_dir?: string;
  dual_tier_ready?: boolean;
  multiclass_loaded?: boolean;
  binary_loaded?: boolean;
  multiclass_model?: string | null;
  binary_model?: string | null;
  binary_model_source?: string;
  severity_model_source?: string;
};

type BackendStatus = {
  state: BackendConnectionState;
  endpoint: string | null;
  message: string;
  checkedAt?: string;
  models?: BackendModelStatus;
};

type QualityReport = {
  is_acceptable: boolean;
  blur_score: number;
  sharpness: number;
  brightness_mean: number;
  contrast_std: number;
  signal_to_noise_ratio: number;
  quality_score: number;
  quality_label: string;
  fundus_area_ratio: number;
  warnings: string[];
  retake_recommendations: string[];
};

type FeatureReport = {
  fundus_area: number;
  vessel_density: number;
  vessel_area: number;
  bright_lesion_area: number;
  dark_lesion_area: number;
  microaneurysm_count: number;
  microaneurysm_area: number;
  exudate_count: number;
  exudate_area: number;
  exudate_quadrants: string[];
  exudate_quadrant_count: number;
  pathology_area_index: number;
  hemorrhage_candidate_count: number;
  optic_disc_area: number;
  optic_disc_detected: boolean;
  mean_intensity: number;
  intensity_std: number;
  texture_contrast: number;
  glcm_contrast: number;
  glcm_homogeneity: number;
  glcm_energy: number;
  expanded_features?: Record<string, number>;
};

type LesionPoint = {
  x: number;
  y: number;
};

type LesionRegion = {
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  centroid: LesionPoint;
  area: number;
  contour: LesionPoint[];
};

type ScreeningTier = {
  status: string;
  referable: boolean;
  rule: string;
  recommendation: string;
};

type ClinicalBasisItem = {
  grade: number | null;
  medical_label: string;
  clinical_reference: string;
  app_mapping: string;
  directly_assessed: boolean;
};

type ScreeningResult = {
  classification: string;
  referable: boolean;
  dr_probability: number;
  stage: number | null;
  stage_label: string;
  medical_label?: string;
  explanation?: string;
  recommendation?: string;
  reason: string;
  disclaimer: string;
  model_type?: string;
  confidence?: number | null;
  confidence_label: string;
  probabilities?: Record<string, number>;
  screening?: ScreeningTier | null;
  screening_recommendation: string;
  consistency_status?: string;
  raw_binary_prediction?: number | null;
  raw_severity_prediction?: number | null;
};

type DetectionFinding = {
  label: string;
  detected: boolean;
};

type AnalysisHistoryEntry = {
  image_id: string;
  date_analyzed: string;
  dr_stage: number | null;
  confidence_level: string;
  screening_recommendation: string;
};

type AnalyzeResponse = {
  filename: string;
  screening_result?: 'referable' | 'non_referable' | 'uncertain' | string;
  screening_label?: string;
  referable_result?: string;
  screening_confidence?: number | null;
  screening_confidence_level?: 'high' | 'medium' | 'low' | string;
  referable_probability?: number | null;
  non_referable_probability?: number | null;
  predicted_class?: number | null;
  severity_grade?: number | null;
  medical_label?: string;
  severity_label_medical?: string;
  grade_confidence?: number | null;
  confidence?: number | null;
  explanation?: string;
  recommendation?: string;
  model_type?: string;
  model_version?: string;
  model_mode?: string;
  binary_model_source?: string;
  severity_model_source?: string;
  consistency_status?: string;
  raw_binary_prediction?: number | null;
  raw_severity_prediction?: number | null;
  clinical_basis?: ClinicalBasisItem[];
  detected_supported_findings?: string[];
  not_directly_assessed_findings?: string[];
  disclaimer?: string;
  clinical_note?: string;
  limitations?: string[];
  model_update_summary?: Record<string, unknown>;
  image_quality_status?:
    | string
    | {
        overall?: string;
        blur?: string;
        brightness?: string;
        contrast?: string;
        warnings?: string[];
        retake_recommendations?: string[];
        quality_score?: number;
        quality_label?: string;
      };
  image_quality?: Record<string, unknown>;
  detected_features?:
    | string[]
    | {
        findings?: string[];
        feature_count?: number;
        expanded_feature_count?: number;
        summary?: Record<string, number>;
      };
  detected_feature_summary?: Record<string, unknown>;
  quality: QualityReport;
  features: FeatureReport;
  result: ScreeningResult;
  processed_images: Record<string, string>;
  detected_findings?: DetectionFinding[];
  history_entry?: AnalysisHistoryEntry | null;
  lesion_regions?: Record<string, LesionRegion[]>;
  image_shape?: Record<string, number>;
};

type AnalyzeTaskResponse = {
  task_id: string;
  status_url: string;
  message: string;
};

type AnalyzeTaskStatusResponse = {
  task_id: string;
  state: 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE' | string;
  message: string;
  result: AnalyzeResponse | null;
  error: string | null;
};

const gallerySaver = NativeModules.DRGallerySaver as
  | GallerySaverModule
  | undefined;
const imageCropper = NativeModules.DRImageCropper as
  | ImageCropModule
  | undefined;
const imagePicker = NativeModules.DRImagePicker as
  | ImagePickerModule
  | undefined;

const PRODUCTION_API_BASE_URL = 'https://optimeye-api-jmogcbpd7a-as.a.run.app';
const LOCAL_NETWORK_API_HOST = '192.168.1.12';
const EXPECTED_MODEL_MODE = 'dual_model_screening_hybrid_severity';
const HEALTH_CHECK_TIMEOUT_MS = __DEV__ ? 4000 : 15000;
const ANALYZE_TIMEOUT_MS = 25000;
const STATUS_POLL_INTERVAL_MS = 1500;
const STATUS_TIMEOUT_MS = 180000;
const ANALYSIS_CROP_SCALE = 1;
const STAGE_OPTIONS = [0, 1, 2, 3, 4];
const palette = {
  canvas: '#F6F9FC',
  surface: '#FFFFFF',
  surfaceTint: '#EEF6F8',
  navy: '#0B2545',
  ink: '#102A43',
  body: '#486581',
  muted: '#627D98',
  line: '#D9E2EC',
  lineStrong: '#BCCCDC',
  primary: '#0B5CAD',
  primaryDark: '#073B78',
  teal: '#127C8A',
  tealDark: '#0B5963',
  tealSoft: '#DDF3F2',
  success: '#198754',
  successSoft: '#E7F6EE',
  warning: '#B7791F',
  warningSoft: '#FFF7E0',
  danger: '#C2414B',
  dangerSoft: '#FFF0F2',
  white: '#FFFFFF',
};
const radius = {
  sm: 10,
  md: 14,
  lg: 18,
  xl: 24,
};
const softShadow = {
  shadowColor: '#0B2545',
  shadowOffset: {width: 0, height: 10},
  shadowOpacity: 0.08,
  shadowRadius: 22,
  elevation: 3,
};
const STAGE_PROBABILITY_ORDER = [
  'No apparent diabetic retinopathy',
  'Mild non-proliferative diabetic retinopathy',
  'Moderate non-proliferative diabetic retinopathy',
  'Severe non-proliferative diabetic retinopathy',
  'Proliferative diabetic retinopathy',
];

const isLoopbackHost = (hostname: string): boolean =>
  ['localhost', '127.0.0.1', '::1'].includes(hostname);

const uniqueStrings = (values: string[]): string[] =>
  values.filter((value, index) => values.indexOf(value) === index);

const getOrderedApiBaseUrls = (preferredUrl: string | null): string[] =>
  uniqueStrings(preferredUrl ? [preferredUrl, ...API_BASE_URLS] : API_BASE_URLS);

const getApiBaseUrlForHost = (protocol: string, hostname: string): string[] => {
  if (Platform.OS === 'android' && isLoopbackHost(hostname)) {
    return [
      `${protocol}//${['10', '0', '2', '2'].join('.')}:8000`,
      `${protocol}//${hostname}:8000`,
    ];
  }

  return [`${protocol}//${hostname}:8000`];
};

const getDevServerApiBaseUrls = (): string[] => {
  const scriptURL = (NativeModules.SourceCode as SourceCodeModule | undefined)
    ?.scriptURL;

  if (!scriptURL) {
    return [];
  }

  try {
    const url = new URL(scriptURL);

    if (!url.hostname) {
      return [];
    }

    return getApiBaseUrlForHost(url.protocol, url.hostname);
  } catch {
    const match = scriptURL.match(/^https?:\/\/([^/:]+)/);
    return match ? getApiBaseUrlForHost('http:', match[1]) : [];
  }
};

const getDevelopmentApiBaseUrls = (): string[] =>
  Platform.OS === 'android'
    ? [
        `http://${['127', '0', '0', '1'].join('.')}:8000`,
        `http://${['10', '0', '2', '2'].join('.')}:8000`,
        ...getDevServerApiBaseUrls(),
        `http://${LOCAL_NETWORK_API_HOST}:8000`,
      ]
    : [
        ...getDevServerApiBaseUrls(),
        ...(Platform.select({
          ios: [`http://${['127', '0', '0', '1'].join('.')}:8000`],
          default: [`http://${['127', '0', '0', '1'].join('.')}:8000`],
        }) ?? []),
        `http://${LOCAL_NETWORK_API_HOST}:8000`,
      ];

const API_BASE_URLS = uniqueStrings(
  !__DEV__ && PRODUCTION_API_BASE_URL
    ? [PRODUCTION_API_BASE_URL]
    : __DEV__
      ? getDevelopmentApiBaseUrls()
      : [],
);

const needsLegacyStoragePermission = (): boolean =>
  Platform.OS === 'android' && Number(Platform.Version) < 29;

const toFileUri = (path: string): string =>
  path.includes('://') ? path : `file://${path}`;

const getFileName = (path?: string): string =>
  path?.split(/[\\/]/).pop()?.split('?')[0] || 'uploaded_fundus.jpg';

const formatNumber = (value: number, decimals = 1): string =>
  Number.isFinite(value) ? value.toFixed(decimals) : '0.0';

export const CLASS_LABELS: Record<number, string> = {
  0: 'No apparent diabetic retinopathy',
  1: 'Mild non-proliferative diabetic retinopathy',
  2: 'Moderate non-proliferative diabetic retinopathy',
  3: 'Severe non-proliferative diabetic retinopathy',
  4: 'Proliferative diabetic retinopathy',
};

export const formatClassValue = (stage: number | null): string =>
  stage === null
    ? 'Not classifiable'
    : CLASS_LABELS[stage] ?? 'Medical severity label unavailable';

const getMedicalLabel = (analysis: AnalyzeResponse): string =>
  analysis.severity_label_medical ||
  analysis.medical_label ||
  analysis.result.medical_label ||
  analysis.result.stage_label ||
  formatClassValue(analysis.result.stage);

const getScreeningLabel = (analysis: AnalyzeResponse): string => {
  if (!analysis.quality.is_acceptable || analysis.screening_result === 'uncertain') {
    return analysis.screening_label || 'Uncertain screening result';
  }
  if (analysis.screening_result === 'referable_review') {
    return 'Referable / Needs ophthalmologist review';
  }
  if (analysis.screening_result === 'referable') {
    return 'Referable DR';
  }
  if (analysis.screening_result === 'non_referable') {
    return 'Non-referable DR';
  }
  if (analysis.screening_label) {
    return analysis.screening_label;
  }
  if (analysis.referable_result) {
    return analysis.referable_result;
  }
  return analysis.result.referable ? 'Referable DR' : 'Non-referable DR';
};

const getScreeningResultKind = (
  analysis: AnalyzeResponse,
): 'referable' | 'non_referable' | 'uncertain' => {
  if (analysis.screening_result === 'referable') {
    return 'referable';
  }

  if (analysis.screening_result === 'referable_review') {
    return 'referable';
  }

  if (analysis.screening_result === 'non_referable') {
    return 'non_referable';
  }

  if (
    analysis.screening_result === 'uncertain' ||
    !analysis.quality.is_acceptable ||
    (analysis.screening_confidence !== null &&
      analysis.screening_confidence !== undefined &&
      analysis.screening_confidence < 0.6)
  ) {
    return 'uncertain';
  }

  return analysis.result.referable ? 'referable' : 'non_referable';
};

const getScreeningConfidencePercent = (
  analysis: AnalyzeResponse,
): number | null => {
  if (!analysis.quality.is_acceptable) {
    return null;
  }
  if (
    analysis.screening_confidence !== null &&
    analysis.screening_confidence !== undefined
  ) {
    return analysis.screening_confidence * 100;
  }
  if (
    analysis.referable_probability !== null &&
    analysis.referable_probability !== undefined &&
    analysis.non_referable_probability !== null &&
    analysis.non_referable_probability !== undefined
  ) {
    return (
      Math.max(
        analysis.referable_probability,
        analysis.non_referable_probability,
      ) * 100
    );
  }
  return null;
};

const getReferableProbabilityPercent = (
  analysis: AnalyzeResponse,
): number | null => {
  if (!analysis.quality.is_acceptable) {
    return null;
  }
  const probability = analysis.referable_probability;
  return probability === null || probability === undefined
    ? null
    : probability * 100;
};

const getScreeningConfidenceLevel = (analysis: AnalyzeResponse): string => {
  const level = analysis.screening_confidence_level;
  if (level) {
    return `${level.charAt(0).toUpperCase()}${level.slice(1)}`;
  }

  const confidence = getScreeningConfidencePercent(analysis);
  if (confidence === null) {
    return 'Unavailable';
  }
  if (confidence >= 80) {
    return 'High';
  }
  if (confidence >= 60) {
    return 'Medium';
  }
  return 'Low';
};

const getGradeConfidencePercent = (analysis: AnalyzeResponse): number | null => {
  const raw = analysis.grade_confidence ?? analysis.result.confidence;
  return raw === null || raw === undefined ? null : raw * 100;
};

const getPlainExplanation = (analysis: AnalyzeResponse): string =>
  analysis.explanation ||
  analysis.result.explanation ||
  analysis.result.reason ||
  'The system analyzed retinal image quality, lesion features, and handcrafted measurements.';

const getRecommendation = (analysis: AnalyzeResponse): string =>
  analysis.recommendation ||
  analysis.result.recommendation ||
  analysis.result.screening_recommendation ||
  getScreeningStatus(analysis).recommendation;

const formatCheckedAt = (): string =>
  new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'Something went wrong.';
};

const isNetworkRequestError = (error: unknown): boolean =>
  [
    'network request failed',
    'request timed out',
    'aborted',
  ].some(fragment =>
    getErrorMessage(error).toLowerCase().includes(fragment),
  );

const isBackendCompatibilityError = (error: unknown): boolean =>
  getErrorMessage(error)
    .toLowerCase()
    .includes('incompatible analysis backend');

const getAnalyzeErrorMessage = (error: unknown): string => {
  const message = getErrorMessage(error);
  const lower = message.toLowerCase();

  if (isBackendCompatibilityError(error)) {
    return 'The analysis server is running an outdated model configuration. Please restart the backend and try again.';
  }

  if (isNetworkRequestError(error)) {
    return 'Unable to connect to analysis server. Please check the backend connection.';
  }

  if (
    lower.includes('readable image') ||
    lower.includes('upload') ||
    lower.includes('image processing failed')
  ) {
    return 'The image could not be analyzed. Please retake it in better focus and lighting, then try again.';
  }

  return 'Analysis could not be completed. Please retake the image or try again.';
};

const isImagePickerCancellation = (error: unknown): boolean => {
  const pickerError = error as { code?: unknown; message?: unknown } | null;
  const code =
    typeof pickerError?.code === 'string' ? pickerError.code.toLowerCase() : '';
  const message = getErrorMessage(error).toLowerCase();

  return (
    code === 'picker_cancelled' ||
    message.includes('no image was selected') ||
    message.includes('picker cancelled')
  );
};

const fetchWithTimeout = async (
  url: string,
  options: RequestInit = {},
  timeoutMs: number,
): Promise<Response> => {
  const controller = new AbortController();
  let didTimeout = false;
  const timeout = setTimeout(() => {
    didTimeout = true;
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    if (didTimeout) {
      throw new Error('Request timed out.');
    }

    throw error;
  } finally {
    clearTimeout(timeout);
  }
};

const sleep = (milliseconds: number): Promise<void> =>
  new Promise(resolve => setTimeout(resolve, milliseconds));

const parseJsonResponse = async (response: Response): Promise<unknown> => {
  const responseText = await response.text();
  return responseText ? JSON.parse(responseText) : null;
};

const isAnalyzeResponse = (value: unknown): value is AnalyzeResponse =>
  typeof value === 'object' &&
  value !== null &&
  'quality' in value &&
  'features' in value &&
  'result' in value;

const isAnalyzeTaskResponse = (value: unknown): value is AnalyzeTaskResponse =>
  typeof value === 'object' &&
  value !== null &&
  'task_id' in value &&
  'status_url' in value;

const getScreeningStatus = (analysis: AnalyzeResponse): ScreeningTier => {
  if (analysis.result.screening) {
    return analysis.result.screening;
  }

  return {
    status: analysis.result.referable ? 'Referable DR' : 'Non-referable DR',
    referable: analysis.result.referable,
    rule: 'Fallback screening mapping from the supporting severity assessment.',
    recommendation: analysis.result.referable
      ? 'Referable diabetic retinopathy detected. Specialist evaluation recommended.'
      : 'No significant referable diabetic retinopathy findings detected. Routine ophthalmology follow-up recommended after clinician review.',
  };
};

const isRuleBasedResult = (result: ScreeningResult): boolean =>
  !result.model_type || result.model_type === 'rule_based';

const formatPercent = (value: number): string =>
  `${formatNumber(value, 1)}%`;

const isHealthResponse = (
  value: unknown,
): value is { status: string; models?: BackendModelStatus } =>
  typeof value === 'object' && value !== null && 'status' in value;

const isExpectedBackendModels = (models?: BackendModelStatus): boolean =>
  models?.model_mode === EXPECTED_MODEL_MODE &&
  models.dual_model_ready === true;

const describeBackendCompatibility = (models?: BackendModelStatus): string => {
  if (!models) {
    return 'model status was not reported';
  }

  const mode = models.model_mode ?? 'unknown mode';
  const readiness = models.dual_model_ready
    ? 'dual models ready'
    : 'dual models not ready';
  return `${mode}; ${readiness}`;
};

const isExpectedAnalysisResponse = (analysis: AnalyzeResponse): boolean =>
  analysis.model_mode === EXPECTED_MODEL_MODE;

const getBackendModelSummary = (models?: BackendModelStatus): string | null => {
  if (!models) {
    return null;
  }

  if (models.dual_model_ready || models.demo_hybrid_ready) {
    return 'Screening and supporting severity models are ready.';
  }

  if (models.dual_tier_ready) {
    return 'Screening models are ready.';
  }

  if (models.multiclass_loaded || models.binary_loaded) {
    const loaded = [
      models.multiclass_loaded ? models.multiclass_model ?? 'grading model' : null,
      models.binary_loaded ? models.binary_model ?? 'screening model' : null,
    ].filter(Boolean);

    return loaded.length > 0
      ? 'Some analysis models are unavailable.'
      : 'Analysis models are unavailable.';
  }

  return 'No trained ML models loaded on the backend.';
};

const pollAnalysisTask = async (
  apiBaseUrl: string,
  taskId: string,
  onProgress: (message: string) => void,
): Promise<AnalyzeResponse> => {
  const startedAt = Date.now();

  while (Date.now() - startedAt < STATUS_TIMEOUT_MS) {
    const response = await fetchWithTimeout(
      `${apiBaseUrl}/status/${taskId}`,
      { method: 'GET' },
      ANALYZE_TIMEOUT_MS,
    );
    const body = (await parseJsonResponse(response)) as
      | AnalyzeTaskStatusResponse
      | null;

    if (!response.ok) {
      const detail =
        typeof (body as { detail?: unknown } | null)?.detail === 'string'
          ? String((body as { detail?: unknown }).detail)
          : 'Task polling failed.';
      throw new Error(detail);
    }

    onProgress(body?.message ?? 'Analysis task is running.');

    if (body?.state === 'SUCCESS' && body.result) {
      return body.result;
    }

    if (body?.state === 'FAILURE') {
      throw new Error(body.error ?? 'Analysis task failed.');
    }

    await sleep(STATUS_POLL_INTERVAL_MS);
  }

  throw new Error('Analysis task timed out while waiting for the backend.');
};

const createAnalyzeFormData = (image: CapturedImage): FormData => {
  const formData = new FormData();
  formData.append(
    'file',
    {
      uri: image.analysisUri,
      name: image.fileName || getFileName(image.analysisPath),
      type: image.mimeType || 'image/jpeg',
    } as unknown as Blob,
  );

  return formData;
};

export default function App(): React.JSX.Element {
  const device = useCameraDevice('back');
  const cameraRef = useRef<CameraRef>(null);
  const photoOutput = usePhotoOutput({
    containerFormat: 'jpeg',
    quality: 0.9,
    qualityPrioritization: 'balanced',
  });

  const { hasPermission, requestPermission } = useCameraPermission();
  const [screen, setScreen] = useState<AppScreen>('home');
  const [isCameraInitialized, setIsCameraInitialized] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isPickingImage, setIsPickingImage] = useState(false);
  const [selectedImage, setSelectedImage] = useState<CapturedImage | null>(
    null,
  );
  const [history, setHistory] = useState<CapturedImage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [hasGalleryPermission, setHasGalleryPermission] = useState(
    () => !needsLegacyStoragePermission(),
  );
  const [activeApiBaseUrl, setActiveApiBaseUrl] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({
    state: 'idle',
    endpoint: null,
    message: 'Connection not checked yet.',
  });
  const [showLesionOverlay, setShowLesionOverlay] = useState(true);
  const [manualStage, setManualStage] = useState<number | null>(null);
  const [overrideConfirmed, setOverrideConfirmed] = useState(false);

  useEffect(() => {
    setShowLesionOverlay(true);
    setManualStage(null);
    setOverrideConfirmed(false);
  }, [selectedImage?.id]);

  const checkGalleryPermission = useCallback(
    async (showDeniedAlert = false): Promise<boolean> => {
      if (!needsLegacyStoragePermission()) {
        setHasGalleryPermission(true);
        return true;
      }

      try {
        const permission = PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE;
        const alreadyGranted = await PermissionsAndroid.check(permission);

        if (alreadyGranted) {
          setHasGalleryPermission(true);
          return true;
        }

        const status = await PermissionsAndroid.request(permission, {
          title: 'Storage Access',
          message:
            'This app needs storage access to save captured images on this Android version.',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        });

        const granted = status === PermissionsAndroid.RESULTS.GRANTED;
        setHasGalleryPermission(granted);

        if (!granted && showDeniedAlert) {
          Alert.alert(
            'Storage Permission Needed',
            'Storage access is needed to save photos on Android 9 and older.',
            [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Open Settings', onPress: Linking.openSettings },
            ],
          );
        }

        return granted;
      } catch (error) {
        console.error('Gallery permission error:', error);
        setHasGalleryPermission(false);
        return false;
      }
    },
    [],
  );

  const savePhotoToGallery = useCallback(
    async (photoPath: string): Promise<string | null> => {
      if (Platform.OS !== 'android') {
        return null;
      }

      if (!gallerySaver) {
        throw new Error(
          'Gallery saver is not installed in the running app. Rebuild and reinstall the Android app.',
        );
      }

      return gallerySaver.saveImage(photoPath, 'DR_Screening');
    },
    [],
  );

  const createAnalysisImage = useCallback(
    async (photoPath: string): Promise<string> => {
      if (Platform.OS !== 'android') {
        return photoPath;
      }

      if (!imageCropper) {
        return photoPath;
      }

      return imageCropper.cropCenterSquare(photoPath, ANALYSIS_CROP_SCALE);
    },
    [],
  );

  const openCapture = useCallback(async () => {
    if (!hasPermission) {
      const granted = await requestPermission();

      if (!granted) {
        Alert.alert(
          'Camera Access Required',
          'Camera permission is required to capture retinal images.',
        );
        return;
      }
    }

    setIsCameraInitialized(false);
    setScreen('capture');
  }, [hasPermission, requestPermission]);

  const applyAnalysisToImage = useCallback(
    (imageId: string, analysis: AnalyzeResponse) => {
      setSelectedImage(current =>
        current?.id === imageId ? { ...current, analysis } : current,
      );
      setHistory(current =>
        current.map(item =>
          item.id === imageId ? { ...item, analysis } : item,
        ),
      );
    },
    [],
  );

  const checkBackendConnection = useCallback(async (): Promise<string | null> => {
    const candidates = getOrderedApiBaseUrls(activeApiBaseUrl);

    setBackendStatus({
      state: 'checking',
      endpoint: activeApiBaseUrl,
      message: 'Checking analysis backend...',
    });

    const probeResults = await Promise.all(
      candidates.map(async apiBaseUrl => {
        try {
          const response = await fetchWithTimeout(
            `${apiBaseUrl}/health`,
            { method: 'GET' },
            HEALTH_CHECK_TIMEOUT_MS,
          );

          if (!response.ok) {
            throw new Error(`Backend health check returned ${response.status}.`);
          }

          const healthBody = await parseJsonResponse(response);
          const models = isHealthResponse(healthBody)
            ? healthBody.models
            : undefined;
          if (!isExpectedBackendModels(models)) {
            throw new Error(
              `Incompatible analysis backend at ${apiBaseUrl}: ${describeBackendCompatibility(
                models,
              )}.`,
            );
          }

          return { apiBaseUrl, models, error: null };
        } catch (error) {
          return { apiBaseUrl, models: undefined, error };
        }
      }),
    );

    const compatible = probeResults.find(result => result.error === null);
    if (compatible) {
      const modelSummary = getBackendModelSummary(compatible.models);
      setActiveApiBaseUrl(compatible.apiBaseUrl);
      setBackendStatus({
        state: 'connected',
        endpoint: compatible.apiBaseUrl,
        message: modelSummary ?? 'Analysis backend is connected.',
        checkedAt: formatCheckedAt(),
        models: compatible.models,
      });
      return compatible.apiBaseUrl;
    }

    const incompatible = probeResults.find(result =>
      isBackendCompatibilityError(result.error),
    );
    const message = incompatible
      ? `${getErrorMessage(
          incompatible.error,
        )} Restart the backend with the current AppDR configuration.`
      : __DEV__
        ? 'Analysis server was not reached. Start the backend on the laptop and confirm the phone and laptop are on the same network.'
        : `Production analysis server was not reached at ${PRODUCTION_API_BASE_URL}. Check the device internet connection and try again.`;

    setBackendStatus({
      state: 'offline',
      endpoint: incompatible?.apiBaseUrl ?? null,
      message,
      checkedAt: formatCheckedAt(),
    });

    return null;
  }, [activeApiBaseUrl]);

  useEffect(() => {
    if (backendStatus.state === 'idle') {
      checkBackendConnection().catch(error => {
        console.error('Backend connection check failed:', error);
      });
    }
  }, [backendStatus.state, checkBackendConnection]);

  const analyzeImage = useCallback(
    async (image: CapturedImage): Promise<void> => {
      setIsAnalyzing(true);
      setAnalysisError(null);
      setManualStage(null);
      setOverrideConfirmed(false);

      try {
        const verifiedApiBaseUrl = await checkBackendConnection();
        if (!verifiedApiBaseUrl) {
          throw new Error(
            'Network request failed: no compatible analysis backend is available.',
          );
        }

        let lastNetworkError: unknown = null;

        for (const apiBaseUrl of getOrderedApiBaseUrls(verifiedApiBaseUrl)) {
          try {
            const response = await fetchWithTimeout(
              `${apiBaseUrl}/analyze`,
              {
                method: 'POST',
                body: createAnalyzeFormData(image),
              },
              ANALYZE_TIMEOUT_MS,
            );

            const responseBody = await parseJsonResponse(response);

            setActiveApiBaseUrl(apiBaseUrl);
            setBackendStatus({
              state: 'connected',
              endpoint: apiBaseUrl,
              message: 'Analysis task submitted.',
              checkedAt: formatCheckedAt(),
            });

            if (!response.ok) {
              const detail =
                typeof (responseBody as { detail?: unknown } | null)?.detail ===
                'string'
                  ? String((responseBody as { detail?: unknown }).detail)
                  : 'Image analysis failed.';
              throw new Error(detail);
            }

            if (isAnalyzeResponse(responseBody)) {
              if (!isExpectedAnalysisResponse(responseBody)) {
                throw new Error(
                  'Incompatible analysis backend: response is not from the defense dual-model configuration.',
                );
              }
              applyAnalysisToImage(image.id, responseBody);
              return;
            }

            if (!isAnalyzeTaskResponse(responseBody)) {
              throw new Error('Backend did not return an analysis task.');
            }

            const analysis = await pollAnalysisTask(
              apiBaseUrl,
              responseBody.task_id,
              message => {
                setBackendStatus({
                  state: 'connected',
                  endpoint: apiBaseUrl,
                  message,
                  checkedAt: formatCheckedAt(),
                });
              },
            );
            if (!isExpectedAnalysisResponse(analysis)) {
              throw new Error(
                'Incompatible analysis backend: response is not from the defense dual-model configuration.',
              );
            }
            applyAnalysisToImage(image.id, analysis);
            return;
          } catch (requestError) {
            if (!isNetworkRequestError(requestError)) {
              throw requestError;
            }

            lastNetworkError = requestError;
          }
        }

        throw lastNetworkError ?? new Error('Network request failed');
      } catch (error) {
        console.error('Analyze error:', error);
        setAnalysisError(getAnalyzeErrorMessage(error));
      } finally {
        setIsAnalyzing(false);
      }
    },
    [applyAnalysisToImage, checkBackendConnection],
  );

  const takePhoto = async (): Promise<void> => {
    if (isCapturing || !isCameraInitialized) {
      return;
    }

    setIsCapturing(true);

    try {
      const photo = await photoOutput.capturePhotoToFile(
        {
          flashMode: 'off',
          enableShutterSound: true,
        },
        {},
      );
      const photoPath = photo.filePath;

      if (!photoPath) {
        throw new Error('No photo file was created.');
      }

      const analysisPath = await createAnalysisImage(photoPath);
      const canSaveToGallery = await checkGalleryPermission(true);
      let savedUri: string | null = null;

      if (canSaveToGallery) {
        try {
          savedUri = await savePhotoToGallery(photoPath);
        } catch (saveError) {
          console.error('Save error:', saveError);
          Alert.alert(
            'Photo Captured',
            `The photo was captured, but saving to the gallery failed.\n\n${getErrorMessage(saveError)}`,
          );
        }
      }

      const capturedImage: CapturedImage = {
        id: `${Date.now()}`,
        path: photoPath,
        uri: toFileUri(photoPath),
        analysisPath,
        analysisUri: toFileUri(analysisPath),
        fileName: getFileName(analysisPath),
        mimeType: 'image/jpeg',
        savedUri,
        savedAt: new Date().toLocaleString(),
        source: 'capture',
      };

      setSelectedImage(capturedImage);
      setHistory(current => [capturedImage, ...current].slice(0, 12));
      setAnalysisError(null);
      setScreen('result');
    } catch (error) {
      console.error('Capture error:', error);
      Alert.alert(
        'Capture Failed',
        getErrorMessage(error) || 'Failed to take photo',
      );
    } finally {
      setIsCapturing(false);
    }
  };

  const uploadImage = useCallback(async (): Promise<void> => {
    if (isPickingImage) {
      return;
    }

    if (Platform.OS !== 'android' || !imagePicker) {
      console.error('DRImagePicker native module is unavailable.');
      Alert.alert('Upload Failed', 'Image upload failed. Please try again.');
      return;
    }

    setIsPickingImage(true);

    try {
      const picked = await imagePicker.pickImage();
      const pickedPath = picked.filePath || picked.fileUri;
      const pickedUri =
        picked.fileUri ||
        (picked.filePath ? toFileUri(picked.filePath) : undefined);

      if (!pickedPath || !pickedUri) {
        throw new Error('Selected image did not include a readable file URI.');
      }

      const uploadedImage: CapturedImage = {
        id: `${Date.now()}`,
        path: pickedPath,
        uri: pickedUri,
        analysisPath: pickedPath,
        analysisUri: pickedUri,
        fileName: picked.name || getFileName(pickedPath),
        mimeType: picked.type || 'image/jpeg',
        savedUri: null,
        savedAt: new Date().toLocaleString(),
        source: 'upload',
      };

      setSelectedImage(uploadedImage);
      setHistory(current => [uploadedImage, ...current].slice(0, 12));
      setAnalysisError(null);
      setScreen('result');
    } catch (error) {
      if (!isImagePickerCancellation(error)) {
        console.error('Upload error:', error);
        Alert.alert('Upload Failed', 'Image upload failed. Please try again.');
      }
    } finally {
      setIsPickingImage(false);
    }
  }, [isPickingImage]);

  const openHistoryImage = useCallback((image: CapturedImage) => {
    setSelectedImage(image);
    setAnalysisError(null);
    setScreen('result');
  }, []);

  const renderTopBar = (title: string, showBack = true) => (
    <View style={styles.topBar}>
      {showBack ? (
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => setScreen('home')}
          activeOpacity={0.75}
        >
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.backPlaceholder} />
      )}
      <Text style={styles.topBarTitle}>{title}</Text>
      <View style={styles.backPlaceholder} />
    </View>
  );

  const renderBackendStatusCard = () => {
    const isChecking = backendStatus.state === 'checking';
    const isConnected = backendStatus.state === 'connected';
    const statusTitle = isConnected
      ? 'Backend connected'
      : isChecking
        ? 'Checking backend'
        : backendStatus.state === 'offline'
          ? 'Backend offline'
          : 'Backend not checked';
    const badgeText = isConnected ? 'OK' : isChecking ? 'CHECK' : 'OFF';

    return (
      <View
        style={[
          styles.connectionPanel,
          isConnected
            ? styles.connectionConnected
            : backendStatus.state === 'offline'
              ? styles.connectionOffline
              : styles.connectionNeutral,
        ]}
      >
        <View style={styles.connectionHeader}>
          <View style={styles.connectionCopy}>
            <Text style={styles.panelEyebrow}>Analysis server</Text>
            <Text style={styles.connectionTitle}>{statusTitle}</Text>
          </View>
          <View
            style={[
              styles.statusBadge,
              isConnected
                ? styles.statusBadgeOk
                : isChecking
                  ? styles.statusBadgeWait
                  : styles.statusBadgeOff,
            ]}
          >
            {isChecking ? (
              <ActivityIndicator color={palette.white} size="small" />
            ) : (
              <Text style={styles.statusBadgeText}>{badgeText}</Text>
            )}
          </View>
        </View>
        <Text style={styles.connectionText}>
          {backendStatus.endpoint ?? backendStatus.message}
        </Text>
        {getBackendModelSummary(backendStatus.models) && (
          <Text style={styles.connectionMeta}>
            {getBackendModelSummary(backendStatus.models)}
          </Text>
        )}
        {backendStatus.checkedAt && (
          <Text style={styles.connectionMeta}>
            Last checked {backendStatus.checkedAt}
          </Text>
        )}
        <TouchableOpacity
          style={[
            styles.compactButton,
            isChecking && styles.compactButtonDisabled,
          ]}
          onPress={checkBackendConnection}
          disabled={isChecking}
          activeOpacity={0.8}
        >
          <Text style={styles.compactButtonText}>
            {isChecking ? 'Checking' : 'Check connection'}
          </Text>
        </TouchableOpacity>
      </View>
    );
  };

  const renderHome = () => (
    <ScrollView style={styles.appSurface} contentContainerStyle={styles.page}>
      <StatusBar barStyle="dark-content" backgroundColor={palette.canvas} />
      <View style={styles.brandBlock}>
        <Text style={styles.brandLabel}>OPTIMEYE</Text>
        <Text style={styles.brandTitle}>Retinal Screening Support</Text>
        <Text style={styles.brandSubtitle}>
          Capture or upload fundus images for quality checks, lesion-supported
          findings, and clinician-reviewable DR screening output.
        </Text>
        <View style={styles.trustStrip}>
          <Text style={styles.trustPill}>Production API</Text>
          <Text style={styles.trustPill}>XGBoost severity support</Text>
          <Text style={styles.trustPill}>203-feature pipeline</Text>
        </View>
      </View>

      <View style={styles.primaryPanel}>
        <View>
          <Text style={styles.panelEyebrow}>Current case</Text>
          <Text style={styles.panelTitle}>
            {selectedImage ? getFileName(selectedImage.path) : 'No image loaded'}
          </Text>
          <Text style={styles.panelText}>
            {selectedImage
              ? selectedImage.savedAt
              : 'Start with a new retinal capture or choose an existing fundus image for analysis.'}
          </Text>
        </View>
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[styles.primaryButton, styles.buttonFlex]}
            onPress={openCapture}
            activeOpacity={0.8}
          >
            <Text style={styles.primaryButtonText}>Start Retinal Capture</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.secondaryButton, styles.buttonFlex]}
            onPress={uploadImage}
            disabled={isPickingImage}
            activeOpacity={0.8}
          >
            <Text style={styles.secondaryButtonText}>
              {isPickingImage ? 'Opening image picker' : 'Upload Fundus Photo'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {renderBackendStatusCard()}

      <View style={styles.grid}>
        <TouchableOpacity
          style={styles.tile}
          onPress={() => setScreen('history')}
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>Screening history</Text>
          <Text style={styles.tileValue}>{history.length}</Text>
          <Text style={styles.tileText}>Saved retinal image records</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.tile}
          onPress={() => setScreen('about')}
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>About DR</Text>
          <Text style={styles.tileValue}>Guide</Text>
          <Text style={styles.tileText}>Clinical context</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.tile}
          onPress={() => setScreen('tips')}
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>Eye Care</Text>
          <Text style={styles.tileValue}>Tips</Text>
          <Text style={styles.tileText}>Patient-friendly reminders</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.tile}
          onPress={() =>
            selectedImage ? setScreen('result') : Alert.alert('No image loaded')
          }
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>Case review</Text>
          <Text style={styles.tileValue}>Case</Text>
          <Text style={styles.tileText}>Open the latest image record</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.noticePanel}>
        <Text style={styles.noticeTitle}>Screening support only</Text>
        <Text style={styles.noticeText}>
          Optimeye supports review; it does not replace diagnosis. A qualified
          eye-care professional should review the image, overlays, and final
          grade before clinical decisions are made.
        </Text>
      </View>
    </ScrollView>
  );

  const renderCapture = () => {
    const isCaptureDisabled =
      !isCameraInitialized || isCapturing;

    if (!device) {
      return (
        <View style={styles.darkCenter}>
          <ActivityIndicator size="large" color={palette.primary} />
          <Text style={styles.loadingText}>Searching for camera...</Text>
          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={() => setScreen('home')}
          >
            <Text style={styles.secondaryButtonText}>Back</Text>
          </TouchableOpacity>
        </View>
      );
    }

    return (
      <View style={styles.captureScreen}>
        <StatusBar barStyle="dark-content" backgroundColor={palette.canvas} />
        <View style={styles.capturePage}>
          <View style={styles.captureHeader}>
            <TouchableOpacity
              style={styles.backButton}
              onPress={() => setScreen('home')}
              activeOpacity={0.75}
            >
              <Text style={styles.backButtonText}>Back</Text>
            </TouchableOpacity>
            <View
              style={[
                styles.captureStatusPill,
                isCameraInitialized && styles.captureStatusReady,
              ]}
            >
              <Text
                style={[
                  styles.captureStatusText,
                  isCameraInitialized && styles.captureStatusTextReady,
                ]}
              >
                {isCameraInitialized ? 'Ready' : 'Starting'}
              </Text>
            </View>
          </View>

          <View style={styles.captureTitleBlock}>
            <Text style={styles.captureEyebrow}>Retinal capture</Text>
            <Text style={styles.captureTitle}>Center the fundus image</Text>
          </View>

          <View style={styles.cameraViewport}>
            <Camera
              ref={cameraRef}
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={screen === 'capture'}
              outputs={[photoOutput]}
              enableNativeZoomGesture={true}
              onStarted={() => setIsCameraInitialized(true)}
              onStopped={() => setIsCameraInitialized(false)}
              onError={error => {
                console.error('Camera error:', error);
                Alert.alert('Camera Error', 'Failed to initialize camera');
              }}
            />
            <View style={styles.cameraFrame} pointerEvents="none">
              <View style={styles.cameraFrameGlow} />
              <View style={styles.retinaTarget}>
                <View style={styles.retinaTargetCore} />
              </View>
              <View style={styles.cornerTopLeft} />
              <View style={styles.cornerTopRight} />
              <View style={styles.cornerBottomLeft} />
              <View style={styles.cornerBottomRight} />
            </View>
          </View>

          <View style={styles.captureGuide}>
            <Text style={styles.captureGuideTitle}>Capture guidance</Text>
            <Text style={styles.captureGuideText}>
              Keep the optic disc and macula inside the square frame. The saved
              crop is sent to the production analysis service.
            </Text>
          </View>

          <View style={styles.captureDock}>
            <TouchableOpacity
              style={[
                styles.captureButton,
                isCaptureDisabled && styles.captureDisabled,
              ]}
              onPress={takePhoto}
              disabled={isCaptureDisabled}
              activeOpacity={0.75}
            >
              {isCapturing ? (
                <ActivityIndicator color={palette.white} size="small" />
              ) : (
                <View style={styles.captureInner} />
              )}
            </TouchableOpacity>
            <Text style={styles.captureCaption}>
              {hasGalleryPermission ? 'Gallery ready' : 'Storage needed'}
            </Text>
          </View>
        </View>
      </View>
    );
  };

  const renderSpecialistOverride = (analysis: AnalyzeResponse) => {
    const systemStage = analysis.result.stage;
    const selectedStage = manualStage ?? systemStage;
    const finalStageLabel =
      selectedStage === null ? 'Not classifiable' : formatClassValue(selectedStage);
    const auditText = overrideConfirmed
      ? `Clinician review recorded locally: ${finalStageLabel}.`
      : 'Awaiting clinician review.';

    return (
      <View style={styles.overridePanel}>
        <Text style={styles.sectionTitle}>Specialist manual review</Text>
        <View style={styles.overrideSummary}>
          <Text style={styles.overrideLabel}>System classification</Text>
          <Text style={styles.overrideValue}>
            {systemStage === null ? 'Not classifiable' : formatClassValue(systemStage)}
          </Text>
        </View>
        <View style={styles.stageSelector}>
          {STAGE_OPTIONS.map(stage => {
            const isSelected = selectedStage === stage;
            return (
              <TouchableOpacity
                key={stage}
                style={[
                  styles.stageOption,
                  isSelected && styles.stageOptionSelected,
                ]}
                onPress={() => {
                  setManualStage(stage);
                  setOverrideConfirmed(false);
                }}
                activeOpacity={0.75}
              >
                <Text
                  style={[
                    styles.stageOptionText,
                    isSelected && styles.stageOptionTextSelected,
                  ]}
                >
                  {CLASS_LABELS[stage]}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
        <View style={styles.actionRow}>
          <TouchableOpacity
            style={[styles.primaryButton, styles.buttonFlex]}
            onPress={() => {
              setManualStage(null);
              setOverrideConfirmed(true);
            }}
            activeOpacity={0.75}
          >
            <Text style={styles.primaryButtonText}>Confirm Assessment</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.secondaryButton, styles.buttonFlex]}
            onPress={() => setOverrideConfirmed(true)}
            activeOpacity={0.75}
          >
            <Text style={styles.secondaryButtonText}>Save Manual Assessment</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.auditText}>{auditText}</Text>
      </View>
    );
  };

  const renderQualityCard = (analysis: AnalyzeResponse) => {
    const retakeRecommendations = analysis.quality.retake_recommendations ?? [];

    return (
      <View
        style={[
          styles.qualityPanel,
          analysis.quality.is_acceptable ? styles.qualityGood : styles.qualityWarn,
        ]}
      >
        <Text style={styles.qualityTitle}>Image Quality</Text>
        <View style={styles.qualityScoreRow}>
          <Text style={styles.qualityScoreValue}>
            {analysis.quality.quality_score} / 100
          </Text>
          <Text style={styles.qualityScoreLabel}>
            {analysis.quality.quality_label}
          </Text>
        </View>
        {retakeRecommendations.length > 0 ? (
          retakeRecommendations.map(message => (
            <Text key={message} style={styles.warningText}>
              {message}
            </Text>
          ))
        ) : (
          <Text style={styles.goodText}>
            Image is suitable for automated screening support.
          </Text>
        )}
      </View>
    );
  };

  const renderFindingsCard = (analysis: AnalyzeResponse) => (
    <View style={styles.summaryPanel}>
      <Text style={styles.sectionTitle}>Detected Findings</Text>
      {(analysis.detected_findings ?? []).map(finding => (
        <View key={finding.label} style={styles.findingRow}>
          <Text
            style={[
              styles.findingMark,
              finding.detected ? styles.findingDetected : styles.findingAbsent,
            ]}
          >
            {finding.detected ? 'Yes' : 'No'}
          </Text>
          <Text style={styles.findingText}>{finding.label}</Text>
        </View>
      ))}
    </View>
  );

  const renderClinicalBasisCard = (analysis: AnalyzeResponse) => {
    const clinicalBasis = analysis.clinical_basis ?? [];

    if (clinicalBasis.length === 0) {
      return null;
    }

    return (
      <View style={styles.summaryPanel}>
        <Text style={styles.sectionTitle}>Clinical Basis</Text>
        {clinicalBasis.map(item => (
          <View key={`${item.grade}-${item.medical_label}`}>
            <Text style={styles.recommendationText}>{item.medical_label}</Text>
            <Text style={styles.reviewDisclaimer}>
              {item.clinical_reference}
            </Text>
            <Text style={styles.reviewDisclaimer}>{item.app_mapping}</Text>
          </View>
        ))}
      </View>
    );
  };

  const renderNotAssessedCard = (analysis: AnalyzeResponse) => {
    const notAssessed = analysis.not_directly_assessed_findings ?? [];

    if (notAssessed.length === 0) {
      return null;
    }

    return (
      <View style={styles.summaryPanel}>
        <Text style={styles.sectionTitle}>Limitations</Text>
        {notAssessed.map(item => (
          <Text key={item} style={styles.reviewDisclaimer}>
            {item}
          </Text>
        ))}
      </View>
    );
  };

  const renderClinicalNoteCard = (analysis: AnalyzeResponse) => (
    <View style={styles.summaryPanel}>
      <Text style={styles.sectionTitle}>Clinical Note</Text>
      <Text style={styles.reviewDisclaimer}>
        {analysis.clinical_note ??
          'This result is for screening support only and is not a final diagnosis. Please confirm with an ophthalmologist.'}
      </Text>
    </View>
  );

  const renderRecommendationCard = (analysis: AnalyzeResponse) => (
    <View style={styles.summaryPanel}>
      <Text style={styles.sectionTitle}>Screening Recommendation</Text>
      <Text style={styles.recommendationText}>
        {getRecommendation(analysis)}
      </Text>
      <Text style={styles.reviewDisclaimer}>
        {analysis.disclaimer || analysis.result.disclaimer}
      </Text>
    </View>
  );

  const renderRuleBasedBanner = (analysis: AnalyzeResponse) => {
    if (!isRuleBasedResult(analysis.result)) {
      return null;
    }

    return (
      <View style={styles.ruleBasedBanner}>
        <Text style={styles.ruleBasedTitle}>Rule-based fallback active</Text>
        <Text style={styles.ruleBasedText}>
          Trained ML models were not used for this result. Ensure the backend is
          running with the trained model artifacts loaded.
        </Text>
      </View>
    );
  };

  const renderProbabilityRow = (label: string, percent: number) => (
    <View key={label} style={styles.probabilityRow}>
      <View style={styles.probabilityLabelRow}>
        <Text style={styles.probabilityLabel}>{label}</Text>
        <Text style={styles.probabilityValue}>{formatPercent(percent)}</Text>
      </View>
      <View style={styles.probabilityTrack}>
        <View
          style={[
            styles.probabilityFill,
            { width: `${Math.min(100, Math.max(0, percent))}%` },
          ]}
        />
      </View>
    </View>
  );

  const renderModelInsightsCard = (analysis: AnalyzeResponse) => {
    const { result } = analysis;

    if (isRuleBasedResult(result)) {
      return null;
    }

    const stageEntries = STAGE_PROBABILITY_ORDER.filter(
      key => result.probabilities?.[key] !== undefined,
    ).map(key => ({
      label: key,
      value: (result.probabilities?.[key] ?? 0) * 100,
    }));

    const screeningEntries = ['Non-Referable', 'Referable']
      .filter(key => result.probabilities?.[key] !== undefined)
      .map(key => ({
        label: key,
        value: (result.probabilities?.[key] ?? 0) * 100,
      }));

    return (
      <View style={styles.summaryPanel}>
        <Text style={styles.sectionTitle}>Screening Details</Text>
        <Text style={styles.modelMetricText}>
          Referable probability: {formatPercent(result.dr_probability)}
        </Text>
        {screeningEntries.length > 0 && (
          <>
            <Text style={styles.probabilityGroupTitle}>Screening probabilities</Text>
            {screeningEntries.map(entry =>
              renderProbabilityRow(entry.label, entry.value),
            )}
          </>
        )}
        {stageEntries.length > 0 && (
          <>
            <Text style={styles.probabilityGroupTitle}>Supporting severity probabilities</Text>
            {stageEntries.map(entry =>
              renderProbabilityRow(entry.label, entry.value),
            )}
          </>
        )}
      </View>
    );
  };

  const renderResult = () => (
    <ScrollView style={styles.appSurface} contentContainerStyle={styles.page}>
      {renderTopBar('Case Review')}
      {selectedImage ? (
        <>
          <Image
            source={{
              uri:
                selectedImage.analysis && showLesionOverlay
                  ? selectedImage.analysis.processed_images.lesion_overlay ??
                    selectedImage.analysis.processed_images.original
                  : selectedImage.analysis?.processed_images.original ??
                    selectedImage.analysisUri,
            }}
            style={styles.preview}
          />
          {selectedImage.analysis && (
            <View style={styles.overlayToggleRow}>
              <View>
                <Text style={styles.overlayToggleTitle}>Lesion overlay</Text>
                <Text style={styles.overlayToggleMeta}>
                  MA and exudate masks from classical processing
                </Text>
              </View>
              <Switch
                value={showLesionOverlay}
                onValueChange={setShowLesionOverlay}
                disabled={!selectedImage.analysis.processed_images.lesion_overlay}
                trackColor={{ false: palette.lineStrong, true: '#8BD8D3' }}
                thumbColor={showLesionOverlay ? palette.teal : palette.white}
              />
            </View>
          )}
          <View
            style={[
              styles.resultBand,
              selectedImage.analysis &&
                (getScreeningResultKind(selectedImage.analysis) === 'referable'
                  ? styles.resultBandReferable
                  : getScreeningResultKind(selectedImage.analysis) === 'non_referable'
                    ? styles.resultBandNonReferable
                    : styles.resultBandUncertain),
            ]}
          >
            <Text style={styles.resultLabel}>
              {isAnalyzing ? 'Analysis running' : 'Screening result'}
            </Text>
            <Text style={styles.resultTitle}>
              {isAnalyzing
                ? 'Automated analysis in progress'
                : selectedImage.analysis
                  ? getScreeningLabel(selectedImage.analysis)
                  : 'Ready to analyze'}
            </Text>
            <Text style={styles.resultText}>
              {isAnalyzing
                ? 'Optimeye is checking image quality, enhancing the retinal image, segmenting vessels, detecting lesions, extracting classical features, and preparing a screening-support result.'
                : selectedImage.analysis
                  ? getPlainExplanation(selectedImage.analysis)
                  : 'Review the selected fundus image, then tap Analyze Image.'}
            </Text>
            {selectedImage.analysis && (
              <View style={styles.resultSummaryRow}>
                <View style={styles.stageBadge}>
                  <Text style={styles.stageBadgeLabel}>Confidence</Text>
                  <Text style={styles.stageBadgeValue}>
                    {getScreeningConfidenceLevel(selectedImage.analysis)}
                  </Text>
                </View>
                <View style={styles.resultFacts}>
                  <Text style={styles.resultFact}>
                    Screening confidence:{' '}
                    {getScreeningConfidencePercent(selectedImage.analysis) === null
                      ? getScreeningConfidenceLevel(selectedImage.analysis)
                      : formatPercent(
                          getScreeningConfidencePercent(selectedImage.analysis) ?? 0,
                        )}
                  </Text>
                  <Text style={styles.resultFact}>
                    Referable probability:{' '}
                    {getReferableProbabilityPercent(selectedImage.analysis) === null
                      ? 'Unavailable'
                      : formatPercent(
                          getReferableProbabilityPercent(selectedImage.analysis) ?? 0,
                        )}
                  </Text>
                  <Text style={styles.resultFact}>
                    Supporting severity assessment: {getMedicalLabel(selectedImage.analysis)}
                    {getGradeConfidencePercent(selectedImage.analysis) !== null
                      ? ` (${formatPercent(
                          getGradeConfidencePercent(selectedImage.analysis) ?? 0,
                        )})`
                      : ''}
                  </Text>
                </View>
              </View>
            )}
          </View>
          {!selectedImage.analysis && (
            <TouchableOpacity
              style={[
                styles.primaryButton,
                styles.analyzeButton,
                isAnalyzing && styles.compactButtonDisabled,
              ]}
              onPress={() => analyzeImage(selectedImage)}
              disabled={isAnalyzing}
              activeOpacity={0.8}
            >
              <Text style={styles.primaryButtonText}>
                {isAnalyzing ? 'Analyzing image' : 'Analyze Image'}
              </Text>
            </TouchableOpacity>
          )}
          {isAnalyzing && (
            <View style={styles.skeletonPanel}>
              <View style={styles.skeletonLineWide} />
              <View style={styles.skeletonLine} />
              <View style={styles.skeletonLineShort} />
            </View>
          )}
          {renderBackendStatusCard()}
          {analysisError && (
            <View style={styles.errorPanel}>
              <Text style={styles.errorPanelTitle}>Analysis failed</Text>
              <Text style={styles.errorPanelText}>{analysisError}</Text>
            </View>
          )}
          {selectedImage.analysis && (
            <>
              {renderRuleBasedBanner(selectedImage.analysis)}
              {renderQualityCard(selectedImage.analysis)}
              {renderFindingsCard(selectedImage.analysis)}
              {renderClinicalBasisCard(selectedImage.analysis)}
              {renderClinicalNoteCard(selectedImage.analysis)}
              {renderNotAssessedCard(selectedImage.analysis)}
              {renderModelInsightsCard(selectedImage.analysis)}
              {renderRecommendationCard(selectedImage.analysis)}

              {selectedImage.analysis.quality.is_acceptable && (
              <View style={styles.processedPanel}>
                <Text style={styles.sectionTitle}>Processed views</Text>
                <View style={styles.processedGrid}>
                  <View style={styles.processedItem}>
                    <Image
                      source={{
                        uri: selectedImage.analysis.processed_images.original,
                      }}
                      style={styles.processedImage}
                    />
                    <Text style={styles.processedLabel}>ROI</Text>
                  </View>
                  <View style={styles.processedItem}>
                    <Image
                      source={{
                        uri: selectedImage.analysis.processed_images.enhanced,
                      }}
                      style={styles.processedImage}
                    />
                    <Text style={styles.processedLabel}>Enhanced</Text>
                  </View>
                  <View style={styles.processedItem}>
                    <Image
                      source={{
                        uri: selectedImage.analysis.processed_images.vessels,
                      }}
                      style={styles.processedImage}
                    />
                    <Text style={styles.processedLabel}>Vessels</Text>
                  </View>
                  <View style={styles.processedItem}>
                    <Image
                      source={{
                        uri: selectedImage.analysis.processed_images
                          .microaneurysms,
                      }}
                      style={styles.processedImage}
                    />
                    <Text style={styles.processedLabel}>MAs</Text>
                  </View>
                  <View style={styles.processedItem}>
                    <Image
                      source={{
                        uri: selectedImage.analysis.processed_images.exudates,
                      }}
                      style={styles.processedImage}
                    />
                    <Text style={styles.processedLabel}>Exudates</Text>
                  </View>
                  <View style={styles.processedItem}>
                    <Image
                      source={{
                        uri: selectedImage.analysis.processed_images
                          .lesion_overlay,
                      }}
                      style={styles.processedImage}
                    />
                    <Text style={styles.processedLabel}>Overlay</Text>
                  </View>
                </View>
              </View>
              )}
            </>
          )}
          <View style={styles.metricGrid}>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Image</Text>
              <Text style={styles.metricValue}>
                {selectedImage.source === 'upload' ? 'Uploaded' : 'Captured'}
              </Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Screening</Text>
              <Text style={styles.metricValue}>
                {selectedImage.analysis
                  ? getScreeningStatus(selectedImage.analysis).status
                  : 'Waiting'}
              </Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Severity assessment</Text>
              <Text style={styles.metricValue}>
                {selectedImage.analysis?.quality.is_acceptable
                  ? getMedicalLabel(selectedImage.analysis)
                  : selectedImage.analysis
                    ? 'N/A'
                    : 'Waiting'}
              </Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Referable risk</Text>
              <Text style={styles.metricValue}>
                {selectedImage.analysis
                  ? getReferableProbabilityPercent(selectedImage.analysis) === null
                    ? 'Unavailable'
                    : formatPercent(
                        getReferableProbabilityPercent(selectedImage.analysis) ?? 0,
                      )
                  : 'Waiting'}
              </Text>
            </View>
          </View>
          {selectedImage.analysis && renderSpecialistOverride(selectedImage.analysis)}
          <Text style={styles.sectionTitle}>Retake or choose another image</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.primaryButton, styles.buttonFlex]}
              onPress={openCapture}
            >
              <Text style={styles.primaryButtonText}>Capture Image</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.buttonFlex]}
              onPress={uploadImage}
              disabled={isPickingImage}
            >
              <Text style={styles.secondaryButtonText}>
                {isPickingImage
                  ? 'Opening image picker'
                  : 'Upload Fundus Photo'}
              </Text>
            </TouchableOpacity>
          </View>
        </>
      ) : (
        <View style={styles.emptyPanel}>
          <Text style={styles.emptyTitle}>No image selected</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.primaryButton, styles.buttonFlex]}
              onPress={openCapture}
            >
              <Text style={styles.primaryButtonText}>Capture Image</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.buttonFlex]}
              onPress={uploadImage}
              disabled={isPickingImage}
            >
              <Text style={styles.secondaryButtonText}>
                Upload Fundus Image
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </ScrollView>
  );

  const renderHistory = () => (
    <ScrollView style={styles.appSurface} contentContainerStyle={styles.page}>
      {renderTopBar('History')}
      {history.length === 0 ? (
        <View style={styles.emptyPanel}>
          <Text style={styles.emptyTitle}>No screening images yet</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.primaryButton, styles.buttonFlex]}
              onPress={openCapture}
            >
              <Text style={styles.primaryButtonText}>Capture Image</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.buttonFlex]}
              onPress={uploadImage}
              disabled={isPickingImage}
            >
              <Text style={styles.secondaryButtonText}>
                Upload Fundus Image
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        history.map(item => (
          <TouchableOpacity
            key={item.id}
            style={styles.historyItem}
            onPress={() => openHistoryImage(item)}
            activeOpacity={0.8}
          >
            <Image source={{ uri: item.analysisUri }} style={styles.historyThumb} />
            <View style={styles.historyTextBlock}>
              <Text style={styles.historyTitle}>{getFileName(item.path)}</Text>
              <Text style={styles.historyMeta}>{item.savedAt}</Text>
              <Text style={styles.historyMeta}>
                {item.source === 'upload'
                  ? 'Uploaded image'
                  : item.savedUri
                    ? 'Saved to gallery'
                    : 'Local capture'}
              </Text>
            </View>
          </TouchableOpacity>
        ))
      )}
    </ScrollView>
  );

  const renderAbout = () => (
    <ScrollView style={styles.appSurface} contentContainerStyle={styles.page}>
      {renderTopBar('About DR')}
      <View style={styles.readingPanel}>
        <Text style={styles.readingTitle}>Diabetic Retinopathy</Text>
        <Text style={styles.readingText}>
          Diabetic retinopathy is a diabetes-related retinal disease caused by
          damage to small blood vessels. Referable cases require professional
          review because they may indicate moderate or advanced retinal changes.
        </Text>
        <Text style={styles.readingTitle}>Dual-Tier ML Screening</Text>
        <Text style={styles.readingText}>
          After classical image processing extracts 203 retinal measurements, a
          trained ML model estimates the diabetic retinopathy grade and a
          binary model screens for referable disease. Output is decision support
          only, not a diagnosis.
        </Text>
        <Text style={styles.readingTitle}>Classical Processing Layer</Text>
        <Text style={styles.readingText}>
          Enhancement, vessel segmentation, lesion detection, and handcrafted
          feature extraction feed the supervised models. When models are
          unavailable, the app falls back to rule-based screening support.
        </Text>
      </View>
    </ScrollView>
  );

  const renderTips = () => (
    <ScrollView style={styles.appSurface} contentContainerStyle={styles.page}>
      {renderTopBar('Eye Care')}
      {[
        'Maintain regular eye screening if diagnosed with diabetes.',
        'Control blood sugar, blood pressure, and cholesterol levels.',
        'Seek professional review when vision changes or new floaters appear.',
        'Use clear, well-focused retinal images for screening support.',
      ].map(tip => (
        <View key={tip} style={styles.tipItem}>
          <View style={styles.tipMarker} />
          <Text style={styles.tipText}>{tip}</Text>
        </View>
      ))}
    </ScrollView>
  );

  const renderScreen = () => {
    switch (screen) {
      case 'capture':
        return renderCapture();
      case 'result':
        return renderResult();
      case 'history':
        return renderHistory();
      case 'about':
        return renderAbout();
      case 'tips':
        return renderTips();
      case 'home':
      default:
        return renderHome();
    }
  };

  return <View style={styles.root}>{renderScreen()}</View>;
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: palette.canvas,
  },
  appSurface: {
    flex: 1,
    backgroundColor: palette.canvas,
  },
  page: {
    padding: 20,
    paddingTop: 48,
    paddingBottom: 38,
  },
  brandBlock: {
    marginBottom: 24,
  },
  brandLabel: {
    color: palette.teal,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  brandTitle: {
    color: palette.navy,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 38,
    marginTop: 8,
  },
  brandSubtitle: {
    color: palette.body,
    fontSize: 15,
    lineHeight: 23,
    marginTop: 10,
  },
  trustStrip: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 16,
  },
  trustPill: {
    borderRadius: 999,
    backgroundColor: palette.tealSoft,
    borderColor: '#B7E3E2',
    borderWidth: 1,
    color: palette.tealDark,
    fontSize: 11,
    fontWeight: '900',
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  primaryPanel: {
    ...softShadow,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    borderRadius: radius.xl,
    padding: 18,
    gap: 18,
    marginBottom: 16,
  },
  panelEyebrow: {
    color: palette.primary,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  panelTitle: {
    color: palette.ink,
    fontSize: 20,
    fontWeight: '900',
    lineHeight: 26,
    marginTop: 6,
  },
  panelText: {
    color: palette.body,
    fontSize: 14,
    lineHeight: 21,
    marginTop: 7,
  },
  primaryButton: {
    minHeight: 54,
    borderRadius: radius.md,
    backgroundColor: palette.primary,
    paddingHorizontal: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  primaryButtonText: {
    color: palette.white,
    fontSize: 15,
    fontWeight: '900',
  },
  buttonRow: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  buttonFlex: {
    flexGrow: 1,
    flexBasis: 160,
  },
  analyzeButton: {
    width: '100%',
    marginBottom: 14,
  },
  secondaryButton: {
    minHeight: 54,
    borderRadius: radius.md,
    backgroundColor: palette.surfaceTint,
    borderColor: '#BBDDE5',
    borderWidth: 1,
    paddingHorizontal: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: palette.primaryDark,
    fontSize: 15,
    fontWeight: '900',
  },
  connectionPanel: {
    ...softShadow,
    borderRadius: radius.lg,
    borderWidth: 1,
    padding: 16,
    marginBottom: 16,
  },
  connectionConnected: {
    backgroundColor: palette.successSoft,
    borderColor: '#A7DCC1',
  },
  connectionOffline: {
    backgroundColor: palette.dangerSoft,
    borderColor: '#F2B8BE',
  },
  connectionNeutral: {
    backgroundColor: palette.surface,
    borderColor: palette.line,
  },
  connectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  connectionCopy: {
    flex: 1,
  },
  connectionTitle: {
    color: palette.ink,
    fontSize: 17,
    fontWeight: '900',
    lineHeight: 22,
    marginTop: 5,
  },
  connectionText: {
    color: palette.body,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 10,
  },
  connectionMeta: {
    color: palette.muted,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 6,
  },
  statusBadge: {
    minWidth: 58,
    height: 34,
    borderRadius: 999,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 11,
  },
  statusBadgeOk: {
    backgroundColor: palette.success,
  },
  statusBadgeWait: {
    backgroundColor: palette.warning,
  },
  statusBadgeOff: {
    backgroundColor: palette.danger,
  },
  statusBadgeText: {
    color: palette.white,
    fontSize: 12,
    fontWeight: '900',
  },
  compactButton: {
    alignSelf: 'flex-start',
    minHeight: 40,
    borderRadius: radius.sm,
    backgroundColor: palette.surface,
    borderColor: palette.lineStrong,
    borderWidth: 1,
    paddingHorizontal: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 12,
  },
  compactButtonDisabled: {
    opacity: 0.7,
  },
  compactButtonText: {
    color: palette.primaryDark,
    fontSize: 13,
    fontWeight: '900',
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  tile: {
    ...softShadow,
    width: '48%',
    minHeight: 138,
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 15,
    justifyContent: 'space-between',
  },
  tileTitle: {
    color: palette.body,
    fontSize: 13,
    fontWeight: '800',
    lineHeight: 18,
  },
  tileValue: {
    color: palette.primary,
    fontSize: 24,
    fontWeight: '900',
  },
  tileText: {
    color: palette.muted,
    fontSize: 12,
    lineHeight: 17,
  },
  noticePanel: {
    marginTop: 18,
    borderRadius: radius.lg,
    backgroundColor: palette.warningSoft,
    borderColor: '#E8D391',
    borderWidth: 1,
    padding: 15,
  },
  noticeTitle: {
    color: palette.warning,
    fontSize: 14,
    fontWeight: '900',
  },
  noticeText: {
    color: '#6F5414',
    fontSize: 13,
    lineHeight: 20,
    marginTop: 6,
  },
  topBar: {
    height: 44,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 18,
  },
  backButton: {
    minWidth: 70,
    height: 40,
    borderRadius: radius.sm,
    backgroundColor: palette.surfaceTint,
    borderColor: '#BBDDE5',
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButtonText: {
    color: palette.primaryDark,
    fontSize: 13,
    fontWeight: '900',
  },
  backPlaceholder: {
    width: 70,
  },
  topBarTitle: {
    color: palette.navy,
    fontSize: 18,
    fontWeight: '900',
  },
  captureScreen: {
    flex: 1,
    backgroundColor: palette.canvas,
  },
  capturePage: {
    flex: 1,
    padding: 20,
    paddingTop: 44,
  },
  captureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  captureTitleBlock: {
    marginTop: 24,
    marginBottom: 18,
  },
  captureEyebrow: {
    color: palette.teal,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  captureTitle: {
    color: palette.navy,
    fontSize: 28,
    fontWeight: '900',
    lineHeight: 34,
    marginTop: 7,
  },
  captureStatusPill: {
    minWidth: 98,
    height: 40,
    borderRadius: 999,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureStatusReady: {
    backgroundColor: palette.successSoft,
    borderColor: '#A7DCC1',
  },
  captureStatusText: {
    color: palette.body,
    fontSize: 13,
    fontWeight: '900',
  },
  captureStatusTextReady: {
    color: palette.success,
  },
  cameraViewport: {
    ...softShadow,
    width: '100%',
    aspectRatio: 1,
    borderRadius: radius.xl,
    backgroundColor: '#D8E7EF',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#ADC8D6',
  },
  cameraFrame: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraFrameGlow: {
    position: 'absolute',
    top: 14,
    right: 14,
    bottom: 14,
    left: 14,
    borderRadius: radius.lg,
    borderColor: 'rgba(255,255,255,0.88)',
    borderWidth: 1,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  retinaTarget: {
    width: '38%',
    aspectRatio: 1,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(11,92,173,0.14)',
  },
  retinaTargetCore: {
    width: '34%',
    aspectRatio: 1,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: 'rgba(255,211,105,0.95)',
  },
  cornerTopLeft: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: 52,
    height: 52,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderColor: palette.white,
  },
  cornerTopRight: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 52,
    height: 52,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderColor: palette.white,
  },
  cornerBottomLeft: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: 52,
    height: 52,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderColor: palette.white,
  },
  cornerBottomRight: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 52,
    height: 52,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderColor: palette.white,
  },
  captureGuide: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    paddingHorizontal: 15,
    paddingVertical: 13,
    marginTop: 16,
  },
  captureGuideTitle: {
    color: palette.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  captureGuideText: {
    color: palette.body,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 5,
  },
  captureDock: {
    alignItems: 'center',
    gap: 12,
    marginTop: 'auto',
    paddingBottom: 12,
  },
  captureButton: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: palette.primary,
    borderColor: '#CFE6F7',
    borderWidth: 7,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureDisabled: {
    backgroundColor: '#91A4B7',
    borderColor: palette.line,
  },
  captureInner: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: palette.white,
  },
  captureCaption: {
    color: palette.body,
    fontSize: 12,
    fontWeight: '800',
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  darkCenter: {
    flex: 1,
    backgroundColor: palette.canvas,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
    padding: 24,
  },
  loadingText: {
    color: palette.body,
    fontSize: 15,
  },
  preview: {
    ...softShadow,
    width: '100%',
    aspectRatio: 1,
    borderRadius: radius.xl,
    backgroundColor: '#D8E7EF',
    marginBottom: 14,
  },
  overlayToggleRow: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    paddingHorizontal: 15,
    paddingVertical: 13,
    marginBottom: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  overlayToggleTitle: {
    color: palette.ink,
    fontSize: 14,
    fontWeight: '900',
  },
  overlayToggleMeta: {
    color: palette.body,
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },
  resultBand: {
    ...softShadow,
    borderRadius: radius.xl,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 18,
    marginBottom: 14,
  },
  resultBandReferable: {
    backgroundColor: palette.dangerSoft,
    borderColor: '#F2B8BE',
  },
  resultBandNonReferable: {
    backgroundColor: palette.successSoft,
    borderColor: '#A7DCC1',
  },
  resultBandUncertain: {
    backgroundColor: palette.warningSoft,
    borderColor: '#E8D391',
  },
  resultLabel: {
    color: palette.teal,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.9,
    textTransform: 'uppercase',
  },
  resultTitle: {
    color: palette.navy,
    fontSize: 21,
    fontWeight: '900',
    lineHeight: 27,
    marginTop: 7,
  },
  resultText: {
    color: palette.body,
    fontSize: 14,
    lineHeight: 21,
    marginTop: 9,
  },
  resultSummaryRow: {
    alignItems: 'stretch',
    flexDirection: 'row',
    gap: 12,
    marginTop: 14,
  },
  stageBadge: {
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.md,
    backgroundColor: palette.primary,
    minWidth: 82,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  stageBadgeLabel: {
    color: '#DCEFFF',
    fontSize: 10,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  stageBadgeValue: {
    color: palette.white,
    fontSize: 18,
    fontWeight: '900',
    marginTop: 2,
  },
  resultFacts: {
    flex: 1,
    gap: 5,
    justifyContent: 'center',
  },
  resultFact: {
    color: palette.tealDark,
    fontSize: 13,
    fontWeight: '800',
    lineHeight: 19,
  },
  skeletonPanel: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
    gap: 10,
  },
  skeletonLineWide: {
    height: 16,
    borderRadius: 999,
    backgroundColor: '#DFEAF2',
    width: '92%',
  },
  skeletonLine: {
    height: 16,
    borderRadius: 999,
    backgroundColor: '#E9F1F6',
    width: '72%',
  },
  skeletonLineShort: {
    height: 16,
    borderRadius: 999,
    backgroundColor: '#F1F6F9',
    width: '45%',
  },
  errorPanel: {
    borderRadius: radius.lg,
    backgroundColor: palette.dangerSoft,
    borderColor: '#F2B8BE',
    borderWidth: 1,
    padding: 15,
    marginBottom: 14,
  },
  errorPanelTitle: {
    color: palette.danger,
    fontSize: 14,
    fontWeight: '900',
  },
  errorPanelText: {
    color: '#7D2F36',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },
  qualityPanel: {
    borderRadius: radius.lg,
    borderWidth: 1,
    padding: 15,
    marginBottom: 14,
  },
  qualityGood: {
    backgroundColor: palette.successSoft,
    borderColor: '#A7DCC1',
  },
  qualityWarn: {
    backgroundColor: palette.warningSoft,
    borderColor: '#E8D391',
  },
  qualityTitle: {
    color: palette.navy,
    fontSize: 16,
    fontWeight: '900',
    marginBottom: 12,
  },
  qualityScoreRow: {
    minHeight: 76,
    borderRadius: radius.md,
    backgroundColor: 'rgba(255,255,255,0.78)',
    paddingHorizontal: 13,
    paddingVertical: 11,
    justifyContent: 'center',
    marginBottom: 10,
  },
  qualityScoreValue: {
    color: palette.navy,
    fontSize: 27,
    fontWeight: '900',
  },
  qualityScoreLabel: {
    color: palette.primaryDark,
    fontSize: 15,
    fontWeight: '900',
    marginTop: 2,
  },
  warningText: {
    color: palette.warning,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },
  goodText: {
    color: palette.success,
    fontSize: 13,
    lineHeight: 19,
  },
  processedPanel: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 15,
    marginBottom: 14,
  },
  sectionTitle: {
    color: palette.navy,
    fontSize: 16,
    fontWeight: '900',
    marginBottom: 12,
  },
  processedGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  processedItem: {
    width: '48%',
    gap: 7,
  },
  processedImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: radius.md,
    backgroundColor: palette.surfaceTint,
  },
  processedLabel: {
    color: palette.body,
    fontSize: 12,
    fontWeight: '800',
    textAlign: 'center',
  },
  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 14,
  },
  metricBox: {
    flexGrow: 1,
    width: '48%',
    minHeight: 88,
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 12,
    justifyContent: 'space-between',
  },
  metricLabel: {
    color: palette.muted,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.3,
    textTransform: 'uppercase',
  },
  metricValue: {
    color: palette.ink,
    fontSize: 14,
    fontWeight: '900',
    lineHeight: 19,
  },
  actionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  summaryPanel: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 15,
    marginBottom: 14,
  },
  findingRow: {
    minHeight: 34,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  findingMark: {
    width: 32,
    fontSize: 12,
    fontWeight: '900',
    textAlign: 'center',
  },
  findingDetected: {
    color: palette.success,
  },
  findingAbsent: {
    color: palette.danger,
  },
  findingText: {
    flex: 1,
    color: palette.ink,
    fontSize: 14,
    fontWeight: '800',
  },
  recommendationText: {
    color: palette.ink,
    fontSize: 15,
    fontWeight: '900',
    lineHeight: 22,
  },
  reviewDisclaimer: {
    color: palette.body,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 10,
  },
  ruleBasedBanner: {
    borderRadius: radius.lg,
    backgroundColor: palette.warningSoft,
    borderColor: '#E8D391',
    borderWidth: 1,
    padding: 15,
    marginBottom: 14,
  },
  ruleBasedTitle: {
    color: palette.warning,
    fontSize: 14,
    fontWeight: '900',
  },
  ruleBasedText: {
    color: '#6F5414',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },
  modelTypeText: {
    color: palette.teal,
    fontSize: 14,
    fontWeight: '900',
    marginTop: 4,
  },
  modelMetricText: {
    color: palette.ink,
    fontSize: 14,
    fontWeight: '800',
    marginTop: 8,
  },
  resultFinePrint: {
    color: palette.body,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 10,
  },
  probabilityGroupTitle: {
    color: palette.body,
    fontSize: 12,
    fontWeight: '900',
    marginTop: 14,
    marginBottom: 8,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  probabilityRow: {
    marginBottom: 10,
  },
  probabilityLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  probabilityLabel: {
    color: palette.ink,
    fontSize: 13,
    fontWeight: '700',
    flex: 1,
  },
  probabilityValue: {
    color: palette.primary,
    fontSize: 13,
    fontWeight: '900',
    marginLeft: 8,
  },
  probabilityTrack: {
    backgroundColor: '#E3EDF5',
    borderRadius: 999,
    height: 8,
    overflow: 'hidden',
  },
  probabilityFill: {
    backgroundColor: palette.primary,
    borderRadius: 999,
    height: 8,
  },
  overridePanel: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 15,
    marginBottom: 14,
  },
  overrideSummary: {
    minHeight: 58,
    borderRadius: radius.md,
    backgroundColor: palette.surfaceTint,
    paddingHorizontal: 13,
    paddingVertical: 11,
    marginBottom: 12,
    alignItems: 'flex-start',
    gap: 6,
  },
  overrideLabel: {
    color: palette.muted,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },
  overrideValue: {
    color: palette.ink,
    fontSize: 14,
    fontWeight: '900',
    lineHeight: 20,
  },
  stageSelector: {
    gap: 8,
    marginBottom: 12,
  },
  stageOption: {
    minHeight: 50,
    borderRadius: radius.md,
    backgroundColor: '#F2F7FA',
    borderColor: palette.line,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  stageOptionSelected: {
    backgroundColor: palette.primary,
    borderColor: palette.primary,
  },
  stageOptionText: {
    color: palette.primaryDark,
    fontSize: 13,
    fontWeight: '900',
    lineHeight: 18,
  },
  stageOptionTextSelected: {
    color: palette.white,
  },
  auditText: {
    color: palette.body,
    fontSize: 12,
    lineHeight: 17,
    marginTop: 10,
  },
  emptyPanel: {
    minHeight: 224,
    borderRadius: radius.xl,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 18,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  emptyTitle: {
    color: palette.navy,
    fontSize: 18,
    fontWeight: '900',
  },
  historyItem: {
    minHeight: 96,
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 11,
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  historyThumb: {
    width: 74,
    height: 74,
    borderRadius: radius.md,
    backgroundColor: palette.surfaceTint,
  },
  historyTextBlock: {
    flex: 1,
    justifyContent: 'center',
  },
  historyTitle: {
    color: palette.ink,
    fontSize: 15,
    fontWeight: '900',
  },
  historyMeta: {
    color: palette.body,
    fontSize: 12,
    lineHeight: 17,
    marginTop: 4,
  },
  readingPanel: {
    borderRadius: radius.xl,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 18,
  },
  readingTitle: {
    color: palette.navy,
    fontSize: 18,
    fontWeight: '900',
    marginBottom: 8,
  },
  readingText: {
    color: palette.body,
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 18,
  },
  tipItem: {
    borderRadius: radius.lg,
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderWidth: 1,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  tipMarker: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: palette.teal,
  },
  tipText: {
    flex: 1,
    color: palette.body,
    fontSize: 14,
    lineHeight: 20,
  },
});

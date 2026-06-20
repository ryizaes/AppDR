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
  filePath: string;
  fileUri: string;
  name: string;
  type: string;
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

const LOCAL_NETWORK_API_BASE_URLS = ['http://192.168.1.12:8000'];
const HEALTH_CHECK_TIMEOUT_MS = 4000;
const ANALYZE_TIMEOUT_MS = 25000;
const STATUS_POLL_INTERVAL_MS = 1500;
const STATUS_TIMEOUT_MS = 180000;
const ANALYSIS_CROP_SCALE = 1;
const STAGE_OPTIONS = [0, 1, 2, 3, 4];
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
    return [`${protocol}//10.0.2.2:8000`, `${protocol}//${hostname}:8000`];
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

const API_BASE_URLS = uniqueStrings(
  Platform.OS === 'android'
    ? [
        'http://127.0.0.1:8000',
        'http://10.0.2.2:8000',
        ...getDevServerApiBaseUrls(),
        ...LOCAL_NETWORK_API_BASE_URLS,
      ]
    : [
        ...getDevServerApiBaseUrls(),
        ...(Platform.select({
          ios: ['http://127.0.0.1:8000'],
          default: ['http://127.0.0.1:8000'],
        }) ?? []),
        ...LOCAL_NETWORK_API_BASE_URLS,
      ],
);

const needsLegacyStoragePermission = (): boolean =>
  Platform.OS === 'android' && Number(Platform.Version) < 29;

const toFileUri = (path: string): string =>
  path.startsWith('file://') ? path : `file://${path}`;

const getFileName = (path: string): string => path.split('/').pop() ?? 'image';

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
  if (analysis.referable_result) {
    return analysis.referable_result;
  }
  if (analysis.screening_result === 'referable_review') {
    return 'Referable / Needs ophthalmologist review';
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
  const raw =
    analysis.screening_confidence ??
    Math.max(
      analysis.referable_probability ?? analysis.result.dr_probability / 100,
      analysis.non_referable_probability ??
        1 - (analysis.result.dr_probability / 100),
    );
  return raw === null || raw === undefined ? null : raw * 100;
};

const getScreeningConfidenceLevel = (analysis: AnalyzeResponse): string => {
  const level = analysis.screening_confidence_level;
  if (level) {
    return `${level.charAt(0).toUpperCase()}${level.slice(1)}`;
  }

  const confidence = getScreeningConfidencePercent(analysis);
  if (confidence === null) {
    return 'Low';
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
  ['network request failed', 'request timed out', 'aborted'].some(fragment =>
    getErrorMessage(error).toLowerCase().includes(fragment),
  );

const getAnalyzeErrorMessage = (error: unknown): string => {
  const message = getErrorMessage(error);
  const lower = message.toLowerCase();

  if (isNetworkRequestError(error)) {
    return `Cannot reach the analysis backend. Tried ${API_BASE_URLS.join(
      ', ',
    )}. Keep FastAPI running on port 8000. For USB testing, keep the phone plugged in and keep the port bridge active.`;
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
      name: getFileName(image.analysisPath),
      type: image.mimeType,
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

  useEffect(() => {
    if (screen === 'capture') {
      checkGalleryPermission();
    }
  }, [checkGalleryPermission, screen]);

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
    let lastError: unknown = null;

    setBackendStatus({
      state: 'checking',
      endpoint: activeApiBaseUrl,
      message: 'Checking analysis backend...',
    });

    for (const apiBaseUrl of candidates) {
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
        const models = isHealthResponse(healthBody) ? healthBody.models : undefined;
        const modelSummary = getBackendModelSummary(models);

        setActiveApiBaseUrl(apiBaseUrl);
        setBackendStatus({
          state: 'connected',
          endpoint: apiBaseUrl,
          message: modelSummary ?? 'Analysis backend is connected.',
          checkedAt: formatCheckedAt(),
          models,
        });
        return apiBaseUrl;
      } catch (error) {
        lastError = error;
      }
    }

    setBackendStatus({
      state: 'offline',
      endpoint: null,
      message: `No backend reached. Last error: ${getErrorMessage(lastError)}`,
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
        let lastNetworkError: unknown = null;

        for (const apiBaseUrl of getOrderedApiBaseUrls(activeApiBaseUrl)) {
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
    [activeApiBaseUrl, applyAnalysisToImage],
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
        mimeType: 'image/jpeg',
        savedUri,
        savedAt: new Date().toLocaleString(),
        source: 'capture',
      };

      setSelectedImage(capturedImage);
      setHistory(current => [capturedImage, ...current].slice(0, 12));
      setScreen('result');
      analyzeImage(capturedImage);
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
      Alert.alert(
        'Upload Not Available',
        'Image upload is currently available in the Android build. Rebuild the app after this update.',
      );
      return;
    }

    setIsPickingImage(true);

    try {
      const picked = await imagePicker.pickImage();
      const uploadedImage: CapturedImage = {
        id: `${Date.now()}`,
        path: picked.filePath,
        uri: picked.fileUri,
        analysisPath: picked.filePath,
        analysisUri: picked.fileUri,
        mimeType: picked.type || 'image/jpeg',
        savedUri: null,
        savedAt: new Date().toLocaleString(),
        source: 'upload',
      };

      setSelectedImage(uploadedImage);
      setHistory(current => [uploadedImage, ...current].slice(0, 12));
      setAnalysisError(null);
      setScreen('result');
      analyzeImage(uploadedImage);
    } catch (error) {
      const message = getErrorMessage(error);

      if (!message.toLowerCase().includes('no image was selected')) {
        Alert.alert('Upload Failed', message);
      }
    } finally {
      setIsPickingImage(false);
    }
  }, [analyzeImage, isPickingImage]);

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
              <ActivityIndicator color="#071014" size="small" />
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
      <StatusBar barStyle="dark-content" backgroundColor="#F5FAF8" />
      <View style={styles.brandBlock}>
        <Text style={styles.brandLabel}>DR Screening</Text>
        <Text style={styles.brandTitle}>Clinician Review Support</Text>
        <Text style={styles.brandSubtitle}>
          Classical retinal image processing with dual-tier supervised ML for
          clinician-reviewed screening support.
        </Text>
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
              : 'Capture a retinal image to begin a screening record.'}
          </Text>
        </View>
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[styles.primaryButton, styles.buttonFlex]}
            onPress={openCapture}
            activeOpacity={0.8}
          >
            <Text style={styles.primaryButtonText}>Capture</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.secondaryButton, styles.buttonFlex]}
            onPress={uploadImage}
            disabled={isPickingImage}
            activeOpacity={0.8}
          >
            <Text style={styles.secondaryButtonText}>
              {isPickingImage ? 'Opening' : 'Upload'}
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
          <Text style={styles.tileTitle}>History</Text>
          <Text style={styles.tileValue}>{history.length}</Text>
          <Text style={styles.tileText}>Saved screening images</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.tile}
          onPress={() => setScreen('about')}
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>About DR</Text>
          <Text style={styles.tileValue}>Info</Text>
          <Text style={styles.tileText}>Clinical context</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.tile}
          onPress={() => setScreen('tips')}
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>Eye Care</Text>
          <Text style={styles.tileValue}>Tips</Text>
          <Text style={styles.tileText}>Retinal health reminders</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.tile}
          onPress={() =>
            selectedImage ? setScreen('result') : Alert.alert('No image loaded')
          }
          activeOpacity={0.8}
        >
          <Text style={styles.tileTitle}>Review</Text>
          <Text style={styles.tileValue}>Case</Text>
          <Text style={styles.tileText}>Latest image record</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.noticePanel}>
        <Text style={styles.noticeTitle}>Screening support only</Text>
        <Text style={styles.noticeText}>
          Screening classifications can under-call advanced disease. A qualified
          eye-care professional must review the image, overlay, and manual grade
          before any clinical decision is made.
        </Text>
      </View>
    </ScrollView>
  );

  const renderCapture = () => {
    const isCaptureDisabled =
      !isCameraInitialized ||
      isCapturing ||
      (needsLegacyStoragePermission() && !hasGalleryPermission);

    if (!device) {
      return (
        <View style={styles.darkCenter}>
          <ActivityIndicator size="large" color="#62D2A2" />
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
        <StatusBar barStyle="dark-content" backgroundColor="#F5FAF8" />
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
            <Text style={styles.captureTitle}>Center Square Scan</Text>
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
            <Text style={styles.captureGuideTitle}>Analysis region</Text>
            <Text style={styles.captureGuideText}>
              The square preview is saved as the analysis image.
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
                <ActivityIndicator color="#FFFFFF" size="small" />
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
                trackColor={{ false: '#CFE3DE', true: '#76D0AE' }}
                thumbColor={showLesionOverlay ? '#0E7C7B' : '#F8FBFA'}
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
                ? 'The system is checking image quality, enhancing the retinal image, segmenting vessels, detecting lesions, extracting classical features, and preparing a screening-support result.'
                : selectedImage.analysis
                  ? getPlainExplanation(selectedImage.analysis)
                  : 'Capture or upload an image to begin automated screening support.'}
            </Text>
            {selectedImage.analysis && (
              <View style={styles.resultSummaryRow}>
                <View style={styles.stageBadge}>
                  <Text style={styles.stageBadgeLabel}>Confidence</Text>
                  <Text style={styles.stageBadgeValue}>
                    {getScreeningConfidenceLevel(selectedImage.analysis)}
                  </Text>
                </View>
                <Text style={styles.resultProbability}>
                  Screening confidence{' '}
                  {getScreeningConfidencePercent(selectedImage.analysis) === null
                    ? getScreeningConfidenceLevel(selectedImage.analysis)
                    : formatPercent(
                        getScreeningConfidencePercent(selectedImage.analysis) ?? 0,
                      )}
                  {'\n'}
                  Referable probability{' '}
                  {formatPercent(
                    (selectedImage.analysis.referable_probability ??
                      selectedImage.analysis.result.dr_probability / 100) * 100,
                  )}
                  {'\n'}
                  Supporting Severity Assessment:{' '}
                  {getMedicalLabel(selectedImage.analysis)}
                  {getGradeConfidencePercent(selectedImage.analysis) !== null
                    ? ` (${formatPercent(
                        getGradeConfidencePercent(selectedImage.analysis) ?? 0,
                      )})`
                    : ''}
                  {'\n'}
                  {getRecommendation(selectedImage.analysis)}
                </Text>
              </View>
            )}
          </View>
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
                  ? formatPercent(selectedImage.analysis.result.dr_probability)
                  : 'Waiting'}
              </Text>
            </View>
          </View>
          {selectedImage.analysis && renderSpecialistOverride(selectedImage.analysis)}
          <View style={styles.actionRow}>
            <TouchableOpacity
              style={[styles.primaryButton, styles.buttonFlex]}
              onPress={openCapture}
            >
              <Text style={styles.primaryButtonText}>Retake</Text>
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
              <Text style={styles.primaryButtonText}>Capture</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.buttonFlex]}
              onPress={uploadImage}
              disabled={isPickingImage}
            >
              <Text style={styles.secondaryButtonText}>Upload</Text>
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
              <Text style={styles.primaryButtonText}>Capture</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.buttonFlex]}
              onPress={uploadImage}
              disabled={isPickingImage}
            >
              <Text style={styles.secondaryButtonText}>Upload</Text>
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
    backgroundColor: '#F5FAF8',
  },
  appSurface: {
    flex: 1,
    backgroundColor: '#F5FAF8',
  },
  page: {
    padding: 20,
    paddingTop: 48,
    paddingBottom: 36,
  },
  brandBlock: {
    marginBottom: 22,
  },
  brandLabel: {
    color: '#0E7C7B',
    fontSize: 13,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  brandTitle: {
    color: '#12323A',
    fontSize: 31,
    fontWeight: '800',
    marginTop: 8,
  },
  brandSubtitle: {
    color: '#5D7378',
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
  },
  primaryPanel: {
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    borderRadius: 8,
    padding: 16,
    gap: 16,
    marginBottom: 16,
    elevation: 2,
  },
  panelEyebrow: {
    color: '#3D8C83',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  panelTitle: {
    color: '#12323A',
    fontSize: 20,
    fontWeight: '800',
    marginTop: 5,
  },
  panelText: {
    color: '#5D7378',
    fontSize: 14,
    marginTop: 6,
  },
  primaryButton: {
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: '#0E7C7B',
    paddingHorizontal: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '800',
  },
  buttonRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 12,
  },
  buttonFlex: {
    flex: 1,
  },
  secondaryButton: {
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: '#EAF4F1',
    borderColor: '#CFE3DE',
    borderWidth: 1,
    paddingHorizontal: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#0E5E63',
    fontSize: 15,
    fontWeight: '800',
  },
  connectionPanel: {
    borderRadius: 8,
    borderWidth: 1,
    padding: 14,
    marginBottom: 16,
  },
  connectionConnected: {
    backgroundColor: '#E8F7F0',
    borderColor: '#A7D8C6',
  },
  connectionOffline: {
    backgroundColor: '#FFF0F1',
    borderColor: '#F0B7BD',
  },
  connectionNeutral: {
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
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
    color: '#12323A',
    fontSize: 17,
    fontWeight: '800',
    marginTop: 5,
  },
  connectionText: {
    color: '#4E666B',
    fontSize: 13,
    lineHeight: 18,
    marginTop: 10,
  },
  connectionMeta: {
    color: '#789096',
    fontSize: 12,
    marginTop: 6,
  },
  statusBadge: {
    minWidth: 54,
    height: 34,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 10,
  },
  statusBadgeOk: {
    backgroundColor: '#3ABF8F',
  },
  statusBadgeWait: {
    backgroundColor: '#F5C96B',
  },
  statusBadgeOff: {
    backgroundColor: '#FF9CA8',
  },
  statusBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  compactButton: {
    alignSelf: 'flex-start',
    minHeight: 38,
    borderRadius: 8,
    backgroundColor: '#F3F9F7',
    borderColor: '#CFE3DE',
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
    color: '#0E5E63',
    fontSize: 13,
    fontWeight: '800',
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  tile: {
    width: '48%',
    minHeight: 132,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 14,
    justifyContent: 'space-between',
  },
  tileTitle: {
    color: '#5D7378',
    fontSize: 13,
    fontWeight: '700',
  },
  tileValue: {
    color: '#0E7C7B',
    fontSize: 24,
    fontWeight: '800',
  },
  tileText: {
    color: '#789096',
    fontSize: 12,
  },
  noticePanel: {
    marginTop: 18,
    borderRadius: 8,
    backgroundColor: '#FFF8E6',
    borderColor: '#E8D59D',
    borderWidth: 1,
    padding: 14,
  },
  noticeTitle: {
    color: '#8A6A12',
    fontSize: 14,
    fontWeight: '800',
  },
  noticeText: {
    color: '#6F622F',
    fontSize: 13,
    lineHeight: 19,
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
    minWidth: 68,
    height: 38,
    borderRadius: 8,
    backgroundColor: '#EAF4F1',
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButtonText: {
    color: '#0E5E63',
    fontSize: 13,
    fontWeight: '800',
  },
  backPlaceholder: {
    width: 68,
  },
  topBarTitle: {
    color: '#12323A',
    fontSize: 18,
    fontWeight: '800',
  },
  captureScreen: {
    flex: 1,
    backgroundColor: '#F5FAF8',
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
    marginTop: 22,
    marginBottom: 18,
  },
  captureEyebrow: {
    color: '#0E7C7B',
    fontSize: 13,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  captureTitle: {
    color: '#12323A',
    fontSize: 27,
    fontWeight: '800',
    marginTop: 6,
  },
  captureStatusPill: {
    minWidth: 96,
    height: 38,
    borderRadius: 8,
    backgroundColor: '#EEF4F2',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureStatusReady: {
    backgroundColor: '#E8F7F0',
    borderColor: '#A7D8C6',
  },
  captureStatusText: {
    color: '#5D7378',
    fontSize: 13,
    fontWeight: '800',
  },
  captureStatusTextReady: {
    color: '#147A5C',
  },
  cameraViewport: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
    backgroundColor: '#DCE8E4',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#BFD8D2',
    elevation: 3,
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
    top: 12,
    right: 12,
    bottom: 12,
    left: 12,
    borderRadius: 8,
    borderColor: 'rgba(255,255,255,0.82)',
    borderWidth: 1,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  retinaTarget: {
    width: '38%',
    aspectRatio: 1,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.85)',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(14,124,123,0.12)',
  },
  retinaTargetCore: {
    width: '34%',
    aspectRatio: 1,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(245,201,107,0.9)',
  },
  cornerTopLeft: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: 52,
    height: 52,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#FFFFFF',
  },
  cornerTopRight: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 52,
    height: 52,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderColor: '#FFFFFF',
  },
  cornerBottomLeft: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: 52,
    height: 52,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#FFFFFF',
  },
  cornerBottomRight: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 52,
    height: 52,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderColor: '#FFFFFF',
  },
  captureGuide: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginTop: 16,
  },
  captureGuideTitle: {
    color: '#0E7C7B',
    fontSize: 13,
    fontWeight: '900',
  },
  captureGuideText: {
    color: '#5D7378',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 4,
  },
  captureDock: {
    alignItems: 'center',
    gap: 12,
    marginTop: 'auto',
    paddingBottom: 12,
  },
  captureButton: {
    width: 86,
    height: 86,
    borderRadius: 43,
    backgroundColor: '#0E7C7B',
    borderColor: '#CFE3DE',
    borderWidth: 7,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureDisabled: {
    backgroundColor: '#94AAA9',
    borderColor: '#D8E8E4',
  },
  captureInner: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: '#FFFFFF',
  },
  captureCaption: {
    color: '#5D7378',
    fontSize: 12,
    fontWeight: '700',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  darkCenter: {
    flex: 1,
    backgroundColor: '#F5FAF8',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
    padding: 24,
  },
  loadingText: {
    color: '#5D7378',
    fontSize: 15,
  },
  preview: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
    backgroundColor: '#DCE8E4',
    marginBottom: 14,
  },
  overlayToggleRow: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginBottom: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  overlayToggleTitle: {
    color: '#12323A',
    fontSize: 14,
    fontWeight: '800',
  },
  overlayToggleMeta: {
    color: '#5D7378',
    fontSize: 12,
    marginTop: 3,
  },
  resultBand: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 16,
    marginBottom: 14,
  },
  resultBandReferable: {
    backgroundColor: '#FFF0F1',
    borderColor: '#F0B7BD',
  },
  resultBandNonReferable: {
    backgroundColor: '#E8F7F0',
    borderColor: '#A7D8C6',
  },
  resultBandUncertain: {
    backgroundColor: '#FFF7E6',
    borderColor: '#E7C36A',
  },
  resultLabel: {
    color: '#0E7C7B',
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  resultTitle: {
    color: '#12323A',
    fontSize: 20,
    fontWeight: '800',
    lineHeight: 25,
    marginTop: 6,
  },
  resultText: {
    color: '#5D7378',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 8,
  },
  resultSummaryRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
  },
  stageBadge: {
    alignItems: 'center',
    borderRadius: 8,
    backgroundColor: '#0E7C7B',
    minWidth: 70,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  stageBadgeLabel: {
    color: '#D9F5EF',
    fontSize: 10,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  stageBadgeValue: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '900',
    marginTop: 2,
  },
  resultProbability: {
    flex: 1,
    color: '#0E5E63',
    fontSize: 15,
    fontWeight: '800',
    lineHeight: 22,
  },
  skeletonPanel: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
    gap: 10,
  },
  skeletonLineWide: {
    height: 16,
    borderRadius: 8,
    backgroundColor: '#E6F0ED',
    width: '92%',
  },
  skeletonLine: {
    height: 16,
    borderRadius: 8,
    backgroundColor: '#EDF5F2',
    width: '72%',
  },
  skeletonLineShort: {
    height: 16,
    borderRadius: 8,
    backgroundColor: '#F3F8F6',
    width: '45%',
  },
  errorPanel: {
    borderRadius: 8,
    backgroundColor: '#FFF0F1',
    borderColor: '#F0B7BD',
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
  },
  errorPanelTitle: {
    color: '#B93A48',
    fontSize: 14,
    fontWeight: '800',
  },
  errorPanelText: {
    color: '#7A3039',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },
  qualityPanel: {
    borderRadius: 8,
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
  },
  qualityGood: {
    backgroundColor: '#E8F7F0',
    borderColor: '#A7D8C6',
  },
  qualityWarn: {
    backgroundColor: '#FFF8E6',
    borderColor: '#E8D59D',
  },
  qualityTitle: {
    color: '#12323A',
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 12,
  },
  qualityScoreRow: {
    minHeight: 72,
    borderRadius: 8,
    backgroundColor: 'rgba(255,255,255,0.72)',
    paddingHorizontal: 12,
    paddingVertical: 10,
    justifyContent: 'center',
    marginBottom: 10,
  },
  qualityScoreValue: {
    color: '#12323A',
    fontSize: 26,
    fontWeight: '900',
  },
  qualityScoreLabel: {
    color: '#0E5E63',
    fontSize: 15,
    fontWeight: '800',
    marginTop: 2,
  },
  warningText: {
    color: '#8A6A12',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },
  goodText: {
    color: '#147A5C',
    fontSize: 13,
    lineHeight: 19,
  },
  processedPanel: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
  },
  sectionTitle: {
    color: '#12323A',
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 12,
  },
  processedGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  processedItem: {
    width: '48%',
    gap: 6,
  },
  processedImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
    backgroundColor: '#EAF4F1',
  },
  processedLabel: {
    color: '#5D7378',
    fontSize: 12,
    fontWeight: '700',
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
    minHeight: 86,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 10,
    justifyContent: 'space-between',
  },
  metricLabel: {
    color: '#789096',
    fontSize: 11,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  metricValue: {
    color: '#12323A',
    fontSize: 14,
    fontWeight: '800',
    lineHeight: 19,
  },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  summaryPanel: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
  },
  findingRow: {
    minHeight: 32,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  findingMark: {
    width: 24,
    fontSize: 17,
    fontWeight: '900',
    textAlign: 'center',
  },
  findingDetected: {
    color: '#147A5C',
  },
  findingAbsent: {
    color: '#9A5660',
  },
  findingText: {
    flex: 1,
    color: '#12323A',
    fontSize: 14,
    fontWeight: '700',
  },
  recommendationText: {
    color: '#12323A',
    fontSize: 15,
    fontWeight: '800',
    lineHeight: 22,
  },
  reviewDisclaimer: {
    color: '#4E666B',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 10,
  },
  ruleBasedBanner: {
    borderRadius: 8,
    backgroundColor: '#FFF8E8',
    borderColor: '#E8C878',
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
  },
  ruleBasedTitle: {
    color: '#8A5A00',
    fontSize: 14,
    fontWeight: '800',
  },
  ruleBasedText: {
    color: '#6B5420',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },
  modelTypeText: {
    color: '#0E7C7B',
    fontSize: 14,
    fontWeight: '800',
    marginTop: 4,
  },
  modelMetricText: {
    color: '#12323A',
    fontSize: 14,
    fontWeight: '700',
    marginTop: 8,
  },
  resultFinePrint: {
    color: '#4E666B',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 10,
  },
  probabilityGroupTitle: {
    color: '#4E666B',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 14,
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  probabilityRow: {
    marginBottom: 10,
  },
  probabilityLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  probabilityLabel: {
    color: '#12323A',
    fontSize: 13,
    fontWeight: '600',
    flex: 1,
  },
  probabilityValue: {
    color: '#0E7C7B',
    fontSize: 13,
    fontWeight: '800',
    marginLeft: 8,
  },
  probabilityTrack: {
    backgroundColor: '#E6F0ED',
    borderRadius: 4,
    height: 8,
    overflow: 'hidden',
  },
  probabilityFill: {
    backgroundColor: '#0E7C7B',
    borderRadius: 4,
    height: 8,
  },
  overridePanel: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 14,
    marginBottom: 14,
  },
  overrideSummary: {
    minHeight: 54,
    borderRadius: 8,
    backgroundColor: '#F6FAF8',
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 12,
    alignItems: 'flex-start',
    gap: 6,
  },
  overrideLabel: {
    color: '#789096',
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  overrideValue: {
    color: '#12323A',
    fontSize: 14,
    fontWeight: '900',
    lineHeight: 20,
  },
  stageSelector: {
    gap: 8,
    marginBottom: 12,
  },
  stageOption: {
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: '#F3F9F7',
    borderColor: '#CFE3DE',
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  stageOptionSelected: {
    backgroundColor: '#0E7C7B',
    borderColor: '#0E7C7B',
  },
  stageOptionText: {
    color: '#0E5E63',
    fontSize: 13,
    fontWeight: '800',
    lineHeight: 18,
  },
  stageOptionTextSelected: {
    color: '#FFFFFF',
  },
  auditText: {
    color: '#5D7378',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 10,
  },
  emptyPanel: {
    minHeight: 220,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 18,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  emptyTitle: {
    color: '#12323A',
    fontSize: 18,
    fontWeight: '800',
  },
  historyItem: {
    minHeight: 92,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 10,
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  historyThumb: {
    width: 72,
    height: 72,
    borderRadius: 8,
    backgroundColor: '#EAF4F1',
  },
  historyTextBlock: {
    flex: 1,
    justifyContent: 'center',
  },
  historyTitle: {
    color: '#12323A',
    fontSize: 15,
    fontWeight: '800',
  },
  historyMeta: {
    color: '#5D7378',
    fontSize: 12,
    marginTop: 4,
  },
  readingPanel: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 16,
  },
  readingTitle: {
    color: '#12323A',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 8,
  },
  readingText: {
    color: '#4E666B',
    fontSize: 14,
    lineHeight: 21,
    marginBottom: 18,
  },
  tipItem: {
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderColor: '#D8E8E4',
    borderWidth: 1,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  tipMarker: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#0E7C7B',
  },
  tipText: {
    flex: 1,
    color: '#4E666B',
    fontSize: 14,
    lineHeight: 20,
  },
});

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  BookOpen,
  CheckCircle2,
  Database,
  FileUp,
  FileText,
  GripVertical,
  Grid2X2,
  FileDown,
  LoaderCircle,
  MoreHorizontal,
  PenSquare,
  PlugZap,
  Plus,
  RefreshCcw,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  Upload,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { ArtifactSurface } from "./ArtifactSurface";
import { DEFAULT_TABULAR_PREPROCESSING_OPTIONS } from "../../lib/backend-types";
import {
  bindRagSource,
  bindOpenProjectSource,
  bindCsvSource,
  bindDbConnectionSource,
  clearSessionSource,
  createDbConnection,
  deleteRagDocument,
  deleteDbConnection,
  exportBoardReport,
  getRagUploadStatus,
  listDbConnectionSchemas,
  listDbConnections,
  listOpenProjectProjects,
  listRagDocuments,
  testDbConnection,
  updateDbConnection,
  uploadRagDocument,
} from "../../lib/backend-api";
import type { BoardExportFormat } from "../../lib/backend-api";
import type {
  ArtifactPayload,
  ChatMessage,
  DBConnection,
  DBConnectionFormPayload,
  DBConnectionSchema,
  DBConnectionType,
  OpenProjectProject,
  OpenProjectSyncResponse,
  RagDocumentStatus,
  SessionSource,
  SessionSourceState,
  TabularPreprocessingOptions,
} from "../../lib/backend-types";
import {
  applyBoardTurnTitleOverrides,
  buildBoardExportSections,
  buildBoardTurnHeaders,
  type BoardTurnTitleOverrides,
  selectVisibleBoardArtifactIds,
} from "../../lib/board-artifacts";
import { summarizeError } from "../../lib/format";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Checkbox } from "../ui/checkbox";

type DashboardTab = "visualizations" | "sources";
type SourceSection = "db" | "csv" | "openproject" | "rag";
type SecretMode = "keep" | "replace" | "clear";
const ARTIFACT_ORDER_KEY = "llm_new_frontend_artifact_order";
const ARTIFACT_HEIGHT_KEY = "llm_new_frontend_artifact_height";
const ARTIFACT_WIDTH_KEY = "llm_new_frontend_artifact_width";
const ARTIFACT_COL_START_KEY = "llm_new_frontend_artifact_col_start";
const BOARD_TURN_TITLE_OVERRIDES_KEY = "llm_new_frontend_board_turn_title_overrides";
const MIN_ARTIFACT_HEIGHT = 220;
const MAX_ARTIFACT_HEIGHT = 800;
const GRID_COLUMNS = 12;
const BOARD_CARD_HEADER_PX = 54;
const BOARD_CARD_BODY_PADDING_PX = 12;
const BOARD_CARD_CODE_PX = 52;
const BOARD_COLUMN_GAP_PX = 12;
const BOARD_GAP_PX = BOARD_COLUMN_GAP_PX;
const BOARD_TURN_HEADER_HEIGHT_PX = 48;
const DEFAULT_ARTIFACT_WIDTH_UNITS = 6;
const MIN_ARTIFACT_WIDTH_UNITS = 4;
const MAX_ARTIFACT_WIDTH_UNITS = 12;

type Props = {
  sessionId: string;
  messages: ChatMessage[];
  artifacts: ArtifactPayload[];
  pinnedArtifactIds: string[];
  userPinnedArtifactIds: string[];
  hiddenArtifactIds: string[];
  /** Increment to switch the left panel to the Visualizations tab. */
  visualizationsFocusBump?: number;
  datasetName: string;
  hasDataset: boolean;
  activeSource: SessionSourceState;
  sources: SessionSource[];
  showCode: boolean;
  onUpload: (files: File[], preprocessingOptions?: TabularPreprocessingOptions) => Promise<void>;
  onRefreshSession: () => Promise<void>;
  onPinArtifactIds: (artifactIds: string[]) => void;
  onUnpinArtifact: (artifactId: string) => void;
};

type UploadItem = {
  name: string;
  date: string;
  status: "analyzed";
  tables: string[];
};

type BooleanPreprocessingOptionKey =
  | "enabled"
  | "detect_csv_separator"
  | "detect_header_row"
  | "normalize_empty_values"
  | "drop_empty_rows"
  | "drop_empty_columns"
  | "drop_sparse_rows"
  | "unique_column_names";

function normalizeRagStatus(status: string | null | undefined): string {
  return String(status || "unknown").replace(/^DocStatus\./, "").toLowerCase();
}

function ragStatusLabel(status: string | null | undefined): string {
  const normalized = normalizeRagStatus(status);
  if (normalized === "processed") {
    return "processed";
  }
  if (normalized === "processing" || normalized === "pending") {
    return "processing";
  }
  if (normalized === "failed" || normalized === "failure") {
    return "failed";
  }
  return normalized;
}

function isRagProcessing(status: string | null | undefined): boolean {
  const normalized = normalizeRagStatus(status);
  return normalized === "processing" || normalized === "pending";
}

function formatRagStatusLabel(status: string | null | undefined): string {
  const normalized = normalizeRagStatus(status);
  if (normalized === "processed") {
    return "обработан";
  }
  if (normalized === "processing" || normalized === "pending") {
    return "обработка";
  }
  if (normalized === "failed" || normalized === "failure") {
    return "ошибка";
  }
  if (normalized === "unknown") {
    return "неизвестно";
  }
  return normalized;
}

type ArtifactBoardLayout = {
  colStart: number;
  widthUnits: number;
  topPx: number;
  heightPx: number;
};

type TurnHeaderBoardLayout = {
  turnKey: string;
  label: string;
  topPx: number;
};

function clampWidthUnitsValue(value: number): number {
  return Math.max(
    MIN_ARTIFACT_WIDTH_UNITS,
    Math.min(MAX_ARTIFACT_WIDTH_UNITS, Math.round(value)),
  );
}

function estimateBoardCardHeight(
  artifact: ArtifactPayload,
  contentHeight: number,
  showCode: boolean,
): number {
  const hasCode =
    showCode && typeof artifact.meta?.code === "string" && artifact.meta.code.length > 0;
  return (
    BOARD_CARD_HEADER_PX +
    BOARD_CARD_BODY_PADDING_PX +
    contentHeight +
    (hasCode ? BOARD_CARD_CODE_PX : 0)
  );
}

function computeBoardLayouts(
  artifacts: ArtifactPayload[],
  turnHeaders: Array<{ turnKey: string; label: string; firstArtifactId: string }>,
  widthMap: Record<string, number>,
  heightMap: Record<string, number>,
  colStartMap: Record<string, number>,
  measuredHeights: Record<string, number>,
  showCode: boolean,
  estimateHeight: (artifact: ArtifactPayload) => number,
): {
  layouts: Map<string, ArtifactBoardLayout>;
  turnHeaderLayouts: TurnHeaderBoardLayout[];
  boardHeight: number;
} {
  const columnBottom = Array.from({ length: GRID_COLUMNS }, () => 0);
  const layouts = new Map<string, ArtifactBoardLayout>();
  const turnHeaderLayouts: TurnHeaderBoardLayout[] = [];
  const headerByFirstArtifactId = new Map(
    turnHeaders.map((header) => [header.firstArtifactId, header]),
  );

  for (const artifact of artifacts) {
    const sectionHeader = headerByFirstArtifactId.get(artifact.id);
    if (sectionHeader) {
      const headerTopPx = Math.max(0, ...columnBottom);
      turnHeaderLayouts.push({
        turnKey: sectionHeader.turnKey,
        label: sectionHeader.label,
        topPx: headerTopPx,
      });
      const belowHeader =
        headerTopPx + BOARD_TURN_HEADER_HEIGHT_PX + BOARD_COLUMN_GAP_PX;
      for (let col = 0; col < GRID_COLUMNS; col += 1) {
        columnBottom[col] = belowHeader;
      }
    }
    const widthUnits = clampWidthUnitsValue(widthMap[artifact.id] ?? DEFAULT_ARTIFACT_WIDTH_UNITS);
    const contentHeight = heightMap[artifact.id] ?? estimateHeight(artifact);
    const heightPx =
      measuredHeights[artifact.id] ??
      estimateBoardCardHeight(artifact, contentHeight, showCode);
    const preferredColStart = colStartMap[artifact.id];

    let colStart = 0;
    let topPx = 0;

    if (preferredColStart != null) {
      colStart = Math.max(
        0,
        Math.min(GRID_COLUMNS - widthUnits, Math.round(preferredColStart) - 1),
      );
      for (let col = colStart; col < colStart + widthUnits; col += 1) {
        topPx = Math.max(topPx, columnBottom[col]);
      }
    } else {
      let bestTop = Number.POSITIVE_INFINITY;
      for (let candidateCol = 0; candidateCol <= GRID_COLUMNS - widthUnits; candidateCol += 1) {
        let candidateTop = 0;
        for (let col = candidateCol; col < candidateCol + widthUnits; col += 1) {
          candidateTop = Math.max(candidateTop, columnBottom[col]);
        }
        if (
          candidateTop < bestTop ||
          (candidateTop === bestTop && candidateCol < colStart)
        ) {
          bestTop = candidateTop;
          colStart = candidateCol;
        }
      }
      topPx = bestTop;
    }

    layouts.set(artifact.id, {
      colStart: colStart + 1,
      widthUnits,
      topPx,
      heightPx,
    });

    const nextBottom = topPx + heightPx + BOARD_COLUMN_GAP_PX;
    for (let col = colStart; col < colStart + widthUnits; col += 1) {
      columnBottom[col] = nextBottom;
    }
  }

  const boardHeight = Math.max(0, ...columnBottom);
  return { layouts, turnHeaderLayouts, boardHeight };
}

function estimateAutoHeight(artifact: ArtifactPayload): number {
  if (artifact.type === "plot") {
    return 440;
  }

  if (artifact.type === "note" && artifact.data.format === "markdown") {
    const content = String((artifact.data.data as { content?: unknown })?.content ?? "");
    const lines = Math.max(1, content.split("\n").length);
    return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, 180 + lines * 18));
  }

  if (artifact.type === "table" && artifact.data.format === "split") {
    const raw = artifact.data.data as { data?: unknown[][]; columns?: unknown[] };
    const rows = Array.isArray(raw.data) ? raw.data.length : 0;
    const cols = Array.isArray(raw.columns) ? raw.columns.length : 0;
    const headerAndPadding = 120;
    const rowHeight = 34;
    return Math.max(
      140,
      Math.min(
        MAX_ARTIFACT_HEIGHT,
        headerAndPadding + Math.min(rows, 20) * rowHeight + Math.min(cols, 12) * 4,
      ),
    );
  }

  if (artifact.type === "value" && artifact.data.format === "value") {
    const data = artifact.data.data as Record<string, unknown>;
    const entries = Object.keys(data ?? {}).length;
    return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, 170 + entries * 40));
  }

  const jsonLength = JSON.stringify(artifact.data.data ?? "").length;
  return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, 200 + Math.min(jsonLength, 4500) / 14));
}

type ConnectionFormState = {
  name: string;
  dbType: DBConnectionType;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  secretMode: SecretMode;
  sslmode: string;
  secure: boolean;
};

type OpenProjectFormState = {
  baseUrl: string;
  apiKey: string;
  project: string;
  days: string;
};

const DEFAULT_FORM: ConnectionFormState = {
  name: "",
  dbType: "postgresql",
  host: "",
  port: "5432",
  database: "",
  username: "",
  password: "",
  secretMode: "replace",
  sslmode: "prefer",
  secure: false,
};

const DEFAULT_OPENPROJECT_FORM: OpenProjectFormState = {
  baseUrl: "http://localhost:8080",
  apiKey: "",
  project: "",
  days: "90",
};

function defaultPortFor(dbType: DBConnectionType): string {
  return dbType === "clickhouse" ? "8123" : "5432";
}

function toFormState(connection?: DBConnection | null): ConnectionFormState {
  if (!connection) {
    return DEFAULT_FORM;
  }
  return {
    name: connection.name,
    dbType: connection.db_type,
    host: connection.host,
    port: connection.port ? String(connection.port) : defaultPortFor(connection.db_type),
    database: connection.database ?? "",
    username: connection.username ?? "",
    password: "",
    secretMode: "keep",
    sslmode:
      connection.db_type === "postgresql"
        ? String(connection.options_json?.sslmode ?? "prefer")
        : "prefer",
    secure: Boolean(connection.options_json?.secure),
  };
}

function buildPayload(form: ConnectionFormState, editing: boolean, existingConnection?: DBConnection | null): DBConnectionFormPayload {
  const existingSchema = typeof existingConnection?.options_json?.schema === "string"
    ? existingConnection.options_json.schema
    : null;
  const payload: DBConnectionFormPayload = {
    name: form.name.trim(),
    db_type: form.dbType,
    host: form.host.trim(),
    port: form.port.trim() ? Number(form.port.trim()) : null,
    database: form.database.trim() || null,
    username: form.username.trim() || null,
    options_json:
      form.dbType === "postgresql"
        ? { sslmode: form.sslmode, schema: existingSchema }
        : { secure: form.secure, schema: existingSchema },
  };

  if (!editing || form.secretMode === "replace") {
    payload.password = form.password || null;
  }
  if (editing && form.secretMode === "clear") {
    payload.clear_password = true;
  }
  return payload;
}

export function DashboardPanel(props: Props) {
  const {
    sessionId,
    messages,
    artifacts,
    pinnedArtifactIds,
    userPinnedArtifactIds,
    hiddenArtifactIds,
    visualizationsFocusBump = 0,
    datasetName,
    hasDataset,
    activeSource,
    sources,
    showCode,
    onUpload,
    onRefreshSession,
    onPinArtifactIds,
    onUnpinArtifact,
  } = props;

  const inputRef = useRef<HTMLInputElement | null>(null);
  const ragInputRef = useRef<HTMLInputElement | null>(null);
  const dbListRef = useRef<HTMLDivElement | null>(null);
  const ragListRef = useRef<HTMLDivElement | null>(null);
  const [activeTab, setActiveTab] = useState<DashboardTab>("visualizations");
  useEffect(() => {
    if (visualizationsFocusBump > 0) {
      setActiveTab("visualizations");
    }
  }, [visualizationsFocusBump]);
  const [sourceSection, setSourceSection] = useState<SourceSection>("db");
  const [recentUploads, setRecentUploads] = useState<UploadItem[]>([]);
  const [ragDocuments, setRagDocuments] = useState<RagDocumentStatus[]>([]);
  const [ragDocumentsLoading, setRagDocumentsLoading] = useState(false);
  const [ragUploading, setRagUploading] = useState(false);
  const [deletingRagDocumentId, setDeletingRagDocumentId] = useState<string | null>(null);
  const [bindingRag, setBindingRag] = useState(false);
  const [bindingOpenProject, setBindingOpenProject] = useState(false);
  const [openProjectSync, setOpenProjectSync] = useState<OpenProjectSyncResponse | null>(null);
  const [openProjectForm, setOpenProjectForm] = useState<OpenProjectFormState>(DEFAULT_OPENPROJECT_FORM);
  const [openProjectProjects, setOpenProjectProjects] = useState<OpenProjectProject[]>([]);
  const [openProjectProjectsLoading, setOpenProjectProjectsLoading] = useState(false);
  const [connections, setConnections] = useState<DBConnection[]>([]);
  const [connectionsLoading, setConnectionsLoading] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[]>([]);
  const [preprocessingOptions, setPreprocessingOptions] =
    useState<TabularPreprocessingOptions>(DEFAULT_TABULAR_PREPROCESSING_OPTIONS);
  const [editingConnection, setEditingConnection] = useState<DBConnection | null>(null);
  const [form, setForm] = useState<ConnectionFormState>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [connectionSchemas, setConnectionSchemas] = useState<Record<string, DBConnectionSchema[]>>({});
  const csvSources = useMemo(
    () => sources.filter((source) => source.source_type === "csv"),
    [sources],
  );
  const csvSourceLabel = useMemo(() => {
    if (csvSources.length === 0) {
      return datasetName;
    }
    if (csvSources.length === 1) {
      return csvSources[0]?.display_name || csvSources[0]?.file_name || datasetName;
    }
    return `${csvSources.length} tabular files`;
  }, [csvSources, datasetName]);
  const [schemasLoadingId, setSchemasLoadingId] = useState<string | null>(null);
  const [bindingId, setBindingId] = useState<string | null>(null);
  const [bindingCsv, setBindingCsv] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearingSource, setClearingSource] = useState(false);
  const [artifactOrderIds, setArtifactOrderIds] = useState<string[]>([]);
  const [draggedArtifactId, setDraggedArtifactId] = useState<string | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const [artifactHeightMap, setArtifactHeightMap] = useState<Record<string, number>>({});
  const [artifactWidthMap, setArtifactWidthMap] = useState<Record<string, number>>({});
  const [artifactColStartMap, setArtifactColStartMap] = useState<Record<string, number>>({});
  const [measuredCardHeights, setMeasuredCardHeights] = useState<Record<string, number>>({});
  const [boardExporting, setBoardExporting] = useState<BoardExportFormat | null>(null);
  const [turnTitleOverrides, setTurnTitleOverrides] = useState<BoardTurnTitleOverrides>({});
  const [renamingTurnHeader, setRenamingTurnHeader] = useState<TurnHeaderBoardLayout | null>(null);
  const [turnTitleDraft, setTurnTitleDraft] = useState("");
  const turnTitleOverridesLoadedSessionRef = useRef("");
  const boardScrollRef = useRef<HTMLDivElement | null>(null);
  const boardCardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const baseArtifacts = artifacts;
  const isOpenProjectMode = activeSource.source_type === "openproject" || activeSource.source_mode === "postgres_sync";
  const artifactById = useMemo(() => {
    const map = new Map<string, ArtifactPayload>();
    baseArtifacts.forEach((artifact) => {
      map.set(artifact.id, artifact);
    });
    return map;
  }, [baseArtifacts]);
  const selectedArtifactIds = useMemo(
    () =>
      selectVisibleBoardArtifactIds({
        artifacts: baseArtifacts,
        messages,
        sessionId,
        autoPinnedArtifactIds: pinnedArtifactIds,
        userPinnedArtifactIds,
        hiddenArtifactIds,
      }),
    [
      baseArtifacts,
      hiddenArtifactIds,
      messages,
      pinnedArtifactIds,
      sessionId,
      userPinnedArtifactIds,
    ],
  );
  const selectedArtifacts = useMemo(
    () => selectedArtifactIds.map((id) => artifactById.get(id)).filter(Boolean) as ArtifactPayload[],
    [artifactById, selectedArtifactIds],
  );
  const visibleConnections = useMemo(
    () =>
      connections.filter((connection) => {
        const options = connection.options_json ?? {};
        return (
          connection.name !== "OpenProject Analytics" &&
          options.source_type !== "openproject" &&
          options.hidden !== true
        );
      }),
    [connections],
  );
  const visibleArtifacts = useMemo(() => {
    const filtered = selectedArtifacts;
    if (filtered.length <= 1) {
      return filtered;
    }
    const byId = new Map(filtered.map((artifact) => [artifact.id, artifact]));
    const ordered: ArtifactPayload[] = [];
    artifactOrderIds.forEach((id) => {
      const artifact = byId.get(id);
      if (artifact) {
        ordered.push(artifact);
        byId.delete(id);
      }
    });
    for (const artifact of filtered) {
      if (byId.has(artifact.id)) {
        ordered.push(artifact);
      }
    }
    return ordered;
  }, [artifactOrderIds, selectedArtifacts]);
  const openProjectReportArtifacts = useMemo(
    () =>
      baseArtifacts.filter(
        (artifact) =>
          artifact.meta?.source_type === "openproject" &&
          artifact.meta?.openproject_report === true,
      ),
    [baseArtifacts],
  );
  const openProjectChartArtifacts = useMemo(
    () => openProjectReportArtifacts.filter((artifact) => artifact.type === "plot"),
    [openProjectReportArtifacts],
  );
  const openProjectTableArtifacts = useMemo(
    () => openProjectReportArtifacts.filter((artifact) => artifact.type === "table"),
    [openProjectReportArtifacts],
  );

  const rawBoardTurnHeaders = useMemo(
    () =>
      buildBoardTurnHeaders(
        visibleArtifacts.map((artifact) => artifact.id),
        baseArtifacts,
        messages,
        sessionId,
      ),
    [baseArtifacts, messages, sessionId, visibleArtifacts],
  );
  const boardTurnHeaders = useMemo(
    () => applyBoardTurnTitleOverrides(rawBoardTurnHeaders, turnTitleOverrides),
    [rawBoardTurnHeaders, turnTitleOverrides],
  );

  const {
    layouts: artifactBoardLayouts,
    turnHeaderLayouts,
    boardHeight: artifactBoardHeight,
  } = useMemo(
    () =>
      computeBoardLayouts(
        visibleArtifacts,
        boardTurnHeaders,
        artifactWidthMap,
        artifactHeightMap,
        artifactColStartMap,
        measuredCardHeights,
        showCode,
        estimateAutoHeight,
      ),
    [
      artifactColStartMap,
      artifactHeightMap,
      artifactWidthMap,
      boardTurnHeaders,
      measuredCardHeights,
      showCode,
      visibleArtifacts,
    ],
  );

  useLayoutEffect(() => {
    const observers: ResizeObserver[] = [];

    visibleArtifacts.forEach((artifact) => {
      const node = boardCardRefs.current.get(artifact.id);
      if (!node) {
        return;
      }

      const observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (!entry) {
          return;
        }
        const nextHeight = Math.ceil(entry.contentRect.height);
        if (nextHeight <= 0) {
          return;
        }
        setMeasuredCardHeights((prev) => {
          if (prev[artifact.id] === nextHeight) {
            return prev;
          }
          return { ...prev, [artifact.id]: nextHeight };
        });
      });

      observer.observe(node);
      observers.push(observer);
    });

    return () => {
      observers.forEach((observer) => observer.disconnect());
    };
  }, [visibleArtifacts, artifactHeightMap, artifactWidthMap, showCode]);

  useEffect(() => {
    setArtifactWidthMap((prev) => {
      let changed = false;
      const next = { ...prev };
      selectedArtifactIds.forEach((id) => {
        if (next[id] != null) {
          return;
        }
        const artifact = artifactById.get(id);
        next[id] = DEFAULT_ARTIFACT_WIDTH_UNITS;
        changed = true;
      });
      return changed ? next : prev;
    });
  }, [artifactById, selectedArtifactIds]);

  const visibleCountRef = useRef(visibleArtifacts.length);
  useEffect(() => {
    if (visibleArtifacts.length > visibleCountRef.current) {
      requestAnimationFrame(() => {
        boardScrollRef.current?.scrollTo({
          top: boardScrollRef.current.scrollHeight,
          behavior: "smooth",
        });
      });
    }
    visibleCountRef.current = visibleArtifacts.length;
  }, [visibleArtifacts.length]);

  const activeSourceText = useMemo(() => {
    if (!activeSource.source_type) {
      return null;
    }
    if (activeSource.source_type === "rag") {
      return {
        label: activeSource.source_label || "База знаний",
        meta: "LightRAG · Документы",
        status: "Готов",
      };
    }
    if (activeSource.source_type === "openproject") {
      return {
        label: activeSource.source_label || "OpenProject",
        meta: "OpenProject · PostgreSQL",
        status: "синхронизирован",
      };
    }
    if (activeSource.source_type === "db_connection") {
      if (activeSource.source_mode === "postgres_sync") {
        return {
          label: activeSource.source_label || "OpenProject",
          meta: "OpenProject · PostgreSQL",
          status: "синхронизирован",
        };
      }
      return {
        label: activeSource.source_label || "Подключение к БД",
        meta: `База данных${activeSource.source_mode ? ` · ${activeSource.source_mode}` : ""}`,
        status: "Подключен",
      };
    }
    return {
      label: csvSourceLabel || activeSource.source_label || "Набор данных CSV/XLSX",
      meta: "CSV/XLSX · Файл загружен",
      status: hasDataset ? "Готов" : "Не загружен",
    };
  }, [activeSource, csvSourceLabel, hasDataset]);

  async function loadConnections(): Promise<void> {
    setConnectionsLoading(true);
    setSourceError(null);
    try {
      setConnections(await listDbConnections());
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setConnectionsLoading(false);
    }
  }

  async function loadRagDocuments(): Promise<void> {
    if (!sessionId) {
      return;
    }
    setRagDocumentsLoading(true);
    setSourceError(null);
    try {
      const response = await listRagDocuments(sessionId);
      setRagDocuments(response.documents);
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setRagDocumentsLoading(false);
    }
  }

  function mergeRagDocuments(nextDocuments: RagDocumentStatus[]): void {
    setRagDocuments((prev) => {
      const byKey = new Map<string, RagDocumentStatus>();
      prev.forEach((document, index) => {
        byKey.set(document.id || document.file_path || `prev-${index}`, document);
      });
      nextDocuments.forEach((document, index) => {
        byKey.set(document.id || document.file_path || `next-${index}`, document);
      });
      return Array.from(byKey.values());
    });
  }

  async function handleDeleteRagDocument(document: RagDocumentStatus): Promise<void> {
    if (!sessionId || !document.id) {
      return;
    }
    const documentId = document.id;
    setDeletingRagDocumentId(documentId);
    setSourceError(null);
    try {
      await deleteRagDocument(sessionId, documentId);
      setRagDocuments((prev) => prev.filter((item) => item.id !== documentId));
      await loadRagDocuments();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setDeletingRagDocumentId(null);
    }
  }

  useEffect(() => {
    if (activeTab === "sources") {
      void loadConnections();
      void loadRagDocuments();
    }
  }, [activeTab]);

  useEffect(() => {
    if (activeSource.source_type === "csv") {
      setSourceSection("csv");
      return;
    }
    if (activeSource.source_type === "rag") {
      setSourceSection("rag");
      return;
    }
    if (activeSource.source_type === "openproject" || activeSource.source_mode === "postgres_sync") {
      setSourceSection("openproject");
      return;
    }
    if (activeSource.source_type === "db_connection") {
      setSourceSection("db");
    }
  }, [activeSource.source_mode, activeSource.source_type]);

  useEffect(() => {
    if (activeSource.source_type !== "csv") {
      return;
    }
    setRecentUploads((prev) => {
      const uploadDate = new Date().toLocaleDateString("ru-RU");
      const uploaded = csvSources.length > 0
        ? csvSources.map((source) => ({
            name: source.display_name || source.file_name || source.alias,
            date: uploadDate,
            status: "analyzed" as const,
            tables: source.csv_table_names ?? [],
          }))
        : datasetName
          ? [{ name: datasetName, date: uploadDate, status: "analyzed" as const, tables: [] }]
          : [];
      const next = [...uploaded, ...prev];
      const deduped = next.filter((item, index, array) => array.findIndex((row) => row.name === item.name) === index);
      return deduped.slice(0, 6);
    });
  }, [activeSource.source_type, csvSources, datasetName]);

  useEffect(() => {
    if (!sessionId) {
      setArtifactOrderIds([]);
      setArtifactHeightMap({});
      setArtifactWidthMap({});
      return;
    }
    const rawOrder = window.localStorage.getItem(`${ARTIFACT_ORDER_KEY}_${sessionId}`);
    if (rawOrder) {
      try {
        const parsed = JSON.parse(rawOrder) as string[];
        setArtifactOrderIds(Array.isArray(parsed) ? parsed : []);
      } catch {
        setArtifactOrderIds([]);
      }
    } else {
      setArtifactOrderIds([]);
    }

    const rawHeight = window.localStorage.getItem(`${ARTIFACT_HEIGHT_KEY}_${sessionId}`);
    if (rawHeight) {
      try {
        const parsed = JSON.parse(rawHeight) as Record<string, number>;
        if (parsed && typeof parsed === "object") {
          const normalized: Record<string, number> = {};
          Object.entries(parsed).forEach(([id, value]) => {
            const numeric = Number(value);
            if (Number.isFinite(numeric)) {
              normalized[id] = Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, numeric));
            }
          });
          setArtifactHeightMap(normalized);
        } else {
          setArtifactHeightMap({});
        }
      } catch {
        setArtifactHeightMap({});
      }
    } else {
      setArtifactHeightMap({});
    }

    const rawWidth = window.localStorage.getItem(`${ARTIFACT_WIDTH_KEY}_${sessionId}`);
    if (rawWidth) {
      try {
        const parsed = JSON.parse(rawWidth) as Record<string, number>;
        if (parsed && typeof parsed === "object") {
          const normalized: Record<string, number> = {};
          Object.entries(parsed).forEach(([id, value]) => {
            const numeric = Number(value);
            if (Number.isFinite(numeric)) {
              normalized[id] = Math.max(
                MIN_ARTIFACT_WIDTH_UNITS,
                Math.min(MAX_ARTIFACT_WIDTH_UNITS, Math.round(numeric)),
              );
            }
          });
          setArtifactWidthMap(normalized);
        } else {
          setArtifactWidthMap({});
        }
      } catch {
        setArtifactWidthMap({});
      }
    } else {
      setArtifactWidthMap({});
    }

    const rawColStart = window.localStorage.getItem(`${ARTIFACT_COL_START_KEY}_${sessionId}`);
    if (rawColStart) {
      try {
        const parsed = JSON.parse(rawColStart) as Record<string, number>;
        if (parsed && typeof parsed === "object") {
          const normalized: Record<string, number> = {};
          Object.entries(parsed).forEach(([id, value]) => {
            const numeric = Number(value);
            if (Number.isFinite(numeric)) {
              normalized[id] = Math.max(1, Math.min(GRID_COLUMNS, Math.round(numeric)));
            }
          });
          setArtifactColStartMap(normalized);
        } else {
          setArtifactColStartMap({});
        }
      } catch {
        setArtifactColStartMap({});
      }
    } else {
      setArtifactColStartMap({});
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      turnTitleOverridesLoadedSessionRef.current = "";
      setTurnTitleOverrides({});
      return;
    }
    const raw = window.localStorage.getItem(`${BOARD_TURN_TITLE_OVERRIDES_KEY}_${sessionId}`);
    if (!raw) {
      turnTitleOverridesLoadedSessionRef.current = sessionId;
      setTurnTitleOverrides({});
      return;
    }
    try {
      const parsed = JSON.parse(raw) as BoardTurnTitleOverrides;
      turnTitleOverridesLoadedSessionRef.current = sessionId;
      setTurnTitleOverrides(
        parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {},
      );
    } catch {
      turnTitleOverridesLoadedSessionRef.current = sessionId;
      setTurnTitleOverrides({});
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${ARTIFACT_ORDER_KEY}_${sessionId}`,
      JSON.stringify(artifactOrderIds),
    );
  }, [artifactOrderIds, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${ARTIFACT_HEIGHT_KEY}_${sessionId}`,
      JSON.stringify(artifactHeightMap),
    );
  }, [artifactHeightMap, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${ARTIFACT_WIDTH_KEY}_${sessionId}`,
      JSON.stringify(artifactWidthMap),
    );
  }, [artifactWidthMap, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${ARTIFACT_COL_START_KEY}_${sessionId}`,
      JSON.stringify(artifactColStartMap),
    );
  }, [artifactColStartMap, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    if (turnTitleOverridesLoadedSessionRef.current !== sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${BOARD_TURN_TITLE_OVERRIDES_KEY}_${sessionId}`,
      JSON.stringify(turnTitleOverrides),
    );
  }, [sessionId, turnTitleOverrides]);

  useEffect(() => {
    const validIds = new Set(baseArtifacts.map((artifact) => artifact.id));
    const sorted = selectedArtifactIds.filter((id) => validIds.has(id));
    setArtifactOrderIds((prev) => {
      if (prev.length === sorted.length && prev.every((id, index) => id === sorted[index])) {
        return prev;
      }
      return sorted;
    });
  }, [baseArtifacts, selectedArtifactIds]);

  useEffect(() => {
    const validIds = new Set(baseArtifacts.map((artifact) => artifact.id));
    setArtifactHeightMap((prev) => {
      const next: Record<string, number> = {};
      Object.entries(prev).forEach(([id, height]) => {
        if (validIds.has(id)) {
          next[id] = height;
        }
      });
      return next;
    });
    setArtifactWidthMap((prev) => {
      const next: Record<string, number> = {};
      Object.entries(prev).forEach(([id, width]) => {
        if (validIds.has(id)) {
          next[id] = width;
        }
      });
      return next;
    });
    setArtifactColStartMap((prev) => {
      const next: Record<string, number> = {};
      Object.entries(prev).forEach(([id, colStart]) => {
        if (validIds.has(id)) {
          next[id] = colStart;
        }
      });
      return next;
    });
    setMeasuredCardHeights((prev) => {
      const next: Record<string, number> = {};
      Object.entries(prev).forEach(([id, height]) => {
        if (validIds.has(id)) {
          next[id] = height;
        }
      });
      return next;
    });
  }, [baseArtifacts]);

  function resetForm(next?: DBConnection | null): void {
    setEditingConnection(next ?? null);
    setForm(toFormState(next ?? null));
    setSourceError(null);
  }

  function resetUploadDialog(): void {
    setPendingUploadFiles([]);
    setPreprocessingOptions(DEFAULT_TABULAR_PREPROCESSING_OPTIONS);
  }

  function updatePreprocessingOption<K extends keyof TabularPreprocessingOptions>(
    key: K,
    value: TabularPreprocessingOptions[K],
  ): void {
    setPreprocessingOptions((prev) => ({ ...prev, [key]: value }));
  }

  function handleFilePick(files: File[]): void {
    if (isOpenProjectMode) {
      return;
    }
    setPendingUploadFiles(files);
    setPreprocessingOptions(DEFAULT_TABULAR_PREPROCESSING_OPTIONS);
    setUploadDialogOpen(true);
  }

  async function handleConfirmUpload(): Promise<void> {
    if (pendingUploadFiles.length === 0) {
      return;
    }
    await onUpload(pendingUploadFiles, preprocessingOptions);
    setUploadDialogOpen(false);
    resetUploadDialog();
    setActiveTab("sources");
  }

  function renderPreprocessingCheckbox(
    key: BooleanPreprocessingOptionKey,
    label: string,
  ) {
    const disabled = key !== "enabled" && !preprocessingOptions.enabled;
    return (
      <label
        key={key}
        className={`flex items-center gap-3 rounded-xl border border-border/50 px-3 py-2 text-sm ${
          disabled ? "text-muted-foreground/50" : "text-foreground"
        }`}
      >
        <Checkbox
          checked={Boolean(preprocessingOptions[key])}
          disabled={disabled}
          onCheckedChange={(checked) => {
            updatePreprocessingOption(key, checked === true);
          }}
        />
        <span className="min-w-0 flex-1">{label}</span>
      </label>
    );
  }

  async function handleOpenSavedConnections(): Promise<void> {
    if (isOpenProjectMode) {
      return;
    }
    setSourceSection("db");
    await loadConnections();
    requestAnimationFrame(() => {
      dbListRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  async function handleOpenKnowledgeBase(): Promise<void> {
    if (isOpenProjectMode) {
      return;
    }
    setSourceSection("rag");
    await loadRagDocuments();
    requestAnimationFrame(() => {
      ragListRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function handleSwitchSource(): void {
    const nextSection =
      activeSource.source_type === "csv"
        ? "csv"
        : activeSource.source_type === "rag"
          ? "rag"
          : activeSource.source_type === "openproject" || activeSource.source_mode === "postgres_sync"
            ? "openproject"
            : "db";
    setSourceSection(nextSection);
    requestAnimationFrame(() => {
      if (nextSection === "db") {
        dbListRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      if (nextSection === "rag") {
        ragListRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  async function handleRagFilePick(file: File): Promise<void> {
    if (!sessionId) {
      return;
    }
    setActiveTab("sources");
    setSourceSection("rag");
    setRagUploading(true);
    setSourceError(null);
    try {
      const upload = await uploadRagDocument(sessionId, file);
      const optimisticDocument: RagDocumentStatus = {
        file_path: file.name,
        status: "processing",
        track_id: upload.track_id,
      };
      mergeRagDocuments([optimisticDocument]);

      for (let attempt = 0; attempt < 45; attempt += 1) {
        const status = await getRagUploadStatus(sessionId, upload.track_id);
        mergeRagDocuments(status.documents.length > 0 ? status.documents : [optimisticDocument]);
        const statuses = status.documents.map((document) => normalizeRagStatus(document.status));
        if (statuses.some((item) => item === "failed" || item === "failure")) {
          break;
        }
        if (statuses.length > 0 && statuses.every((item) => item === "processed")) {
          break;
        }
        await new Promise((resolve) => {
          window.setTimeout(resolve, 2000);
        });
      }

      await loadRagDocuments();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setRagUploading(false);
    }
  }

  async function handleSubmitConnection(): Promise<void> {
    setSubmitting(true);
    setSourceError(null);
    try {
      const payload = buildPayload(form, Boolean(editingConnection), editingConnection);
      if (editingConnection) {
        await updateDbConnection(editingConnection.id, payload);
      } else {
        await createDbConnection(payload);
      }
      setDialogOpen(false);
      resetForm(null);
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDeleteConnection(connection: DBConnection): Promise<void> {
    if (!window.confirm(`Удалить подключение "${connection.name}"?`)) {
      return;
    }
    setDeletingId(connection.id);
    setSourceError(null);
    try {
      await deleteDbConnection(connection.id);
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setDeletingId(null);
    }
  }

  async function handleTestConnection(connection: DBConnection): Promise<void> {
    setTestingId(connection.id);
    try {
      const result = await testDbConnection(connection.id);
      await loadConnections();
      if (result.last_test_ok) {
        void loadSchemasForConnection(connection.id);
      }
    } catch (error) {
      await loadConnections();
    } finally {
      setTestingId(null);
    }
  }

  async function loadSchemasForConnection(connectionId: string): Promise<void> {
    setSchemasLoadingId(connectionId);
    try {
      const schemas = await listDbConnectionSchemas(connectionId);
      setConnectionSchemas((prev) => ({ ...prev, [connectionId]: schemas }));
    } catch {
      // ignore — schemas will remain unavailable
    } finally {
      setSchemasLoadingId((prev) => (prev === connectionId ? null : prev));
    }
  }

  async function handleSelectSchema(connection: DBConnection, schema: string): Promise<void> {
    try {
      const updatedOptionsJson = { ...(connection.options_json ?? {}), schema: schema || null };
      await updateDbConnection(connection.id, { options_json: updatedOptionsJson });
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    }
  }

  async function handleBindConnection(connection: DBConnection): Promise<void> {
    if (!sessionId || isOpenProjectMode) {
      return;
    }
    setBindingId(connection.id);
    setSourceError(null);
    try {
      await bindDbConnectionSource(sessionId, connection.id);
      await onRefreshSession();
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBindingId(null);
    }
  }

  async function handleClearSource(): Promise<void> {
    if (!sessionId || isOpenProjectMode) {
      return;
    }
    setClearingSource(true);
    setSourceError(null);
    try {
      await clearSessionSource(sessionId);
      await onRefreshSession();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setClearingSource(false);
    }
  }

  async function handleBindCsvSource(): Promise<void> {
    if (!sessionId || !hasDataset || !datasetName || isOpenProjectMode) {
      return;
    }
    setBindingCsv(true);
    setSourceError(null);
    try {
      await bindCsvSource(sessionId);
      await onRefreshSession();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBindingCsv(false);
    }
  }

  async function handleBindRagSource(): Promise<void> {
    if (!sessionId || isOpenProjectMode) {
      return;
    }
    setBindingRag(true);
    setSourceError(null);
    try {
      await bindRagSource(sessionId);
      await onRefreshSession();
      await loadRagDocuments();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBindingRag(false);
    }
  }

  async function handleBindOpenProjectSource(): Promise<void> {
    if (!sessionId) {
      return;
    }
    setBindingOpenProject(true);
    setSourceError(null);
    try {
      const result = await bindOpenProjectSource(sessionId, buildOpenProjectPayload());
      setOpenProjectSync(result);
      await onRefreshSession();
      onPinArtifactIds(result.artifact_ids);
      setActiveTab("visualizations");
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBindingOpenProject(false);
    }
  }

  function buildOpenProjectPayload() {
    const days = Number.parseInt(openProjectForm.days.trim(), 10);
    return {
      base_url: openProjectForm.baseUrl.trim() || null,
      api_key: openProjectForm.apiKey.trim() || null,
      project: openProjectForm.project.trim() || null,
      all_projects: !openProjectForm.project.trim(),
      days: Number.isFinite(days) ? days : null,
    };
  }

  async function handleLoadOpenProjectProjects(): Promise<void> {
    if (!sessionId) {
      return;
    }
    setOpenProjectProjectsLoading(true);
    setSourceError(null);
    try {
      const result = await listOpenProjectProjects(sessionId, buildOpenProjectPayload());
      setOpenProjectProjects(result.projects);
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setOpenProjectProjectsLoading(false);
    }
  }

  function handleDropAtIndex(targetIndex: number): void {
    if (!draggedArtifactId) {
      return;
    }

    const currentIds = visibleArtifacts.map((artifact) => artifact.id);
    const fromIndex = currentIds.indexOf(draggedArtifactId);
    if (fromIndex === -1) {
      setDraggedArtifactId(null);
      setDropIndex(null);
      return;
    }

    const nextIds = [...currentIds];
    nextIds.splice(fromIndex, 1);
    const normalizedTarget = Math.max(0, Math.min(targetIndex, nextIds.length));
    nextIds.splice(normalizedTarget, 0, draggedArtifactId);

    setArtifactOrderIds((prev) => {
      const tail = prev.filter((id) => !nextIds.includes(id));
      return [...nextIds, ...tail];
    });
    setArtifactColStartMap({});
    setDraggedArtifactId(null);
    setDropIndex(null);
  }

  function clampWidthUnits(value: number): number {
    return Math.max(
      MIN_ARTIFACT_WIDTH_UNITS,
      Math.min(MAX_ARTIFACT_WIDTH_UNITS, Math.round(value)),
    );
  }

  function clampHeightPx(value: number): number {
    return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, Math.round(value)));
  }

  function startPointerResize(
    artifact: ArtifactPayload,
    mode: "width" | "height" | "both" | "width-left",
    event: MouseEvent,
  ): void {
    event.preventDefault();
    event.stopPropagation();
    const initialX = event.clientX;
    const initialY = event.clientY;
    const initialWidth = artifactWidthMap[artifact.id] ?? DEFAULT_ARTIFACT_WIDTH_UNITS;
    const initialHeight = artifactHeightMap[artifact.id] ?? estimateAutoHeight(artifact);
    const initialLayout = artifactBoardLayouts.get(artifact.id);
    const initialColStart =
      artifactColStartMap[artifact.id] ?? initialLayout?.colStart ?? 1;

    const handleMove = (moveEvent: MouseEvent) => {
      if (mode === "width" || mode === "both") {
        const deltaX = moveEvent.clientX - initialX;
        const nextWidth = clampWidthUnits(initialWidth + deltaX / 72);
        setArtifactWidthMap((prev) => ({ ...prev, [artifact.id]: nextWidth }));
        setArtifactColStartMap({ [artifact.id]: initialColStart });
      }
      if (mode === "width-left") {
        const deltaX = moveEvent.clientX - initialX;
        const widthDelta = Math.round(deltaX / 72);
        const nextWidth = clampWidthUnits(initialWidth - widthDelta);
        const rightEdge = initialColStart + initialWidth - 1;
        const nextColStart = Math.max(1, rightEdge - nextWidth + 1);
        setArtifactWidthMap((prev) => ({ ...prev, [artifact.id]: nextWidth }));
        setArtifactColStartMap({ [artifact.id]: nextColStart });
      }
      if (mode === "height" || mode === "both") {
        const deltaY = moveEvent.clientY - initialY;
        const nextHeight = clampHeightPx(initialHeight + deltaY);
        setArtifactHeightMap((prev) => ({ ...prev, [artifact.id]: nextHeight }));
      }
    };

    const handleUp = () => {
      document.removeEventListener("mousemove", handleMove);
      document.removeEventListener("mouseup", handleUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.userSelect = "none";
    document.body.style.cursor =
      mode === "both"
        ? "nwse-resize"
        : mode === "width" || mode === "width-left"
          ? "ew-resize"
          : "ns-resize";
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp, { once: true });
  }

  function resetArtifactSize(artifact: ArtifactPayload): void {
    setArtifactWidthMap((prev) => ({ ...prev, [artifact.id]: DEFAULT_ARTIFACT_WIDTH_UNITS }));
    setArtifactHeightMap((prev) => ({ ...prev, [artifact.id]: estimateAutoHeight(artifact) }));
    setArtifactColStartMap((prev) => {
      const next = { ...prev };
      delete next[artifact.id];
      return next;
    });
  }

  const boardExportTitle = useMemo(() => {
    if (boardTurnHeaders.length === 1) {
      return (
        boardTurnHeaders[0].label.replace(/^Вопрос\s+\d+:\s*/i, "").trim() ||
        "Отчёт по визуализациям"
      );
    }
    if (boardTurnHeaders.length > 1) {
      return "Аналитический отчёт по доске";
    }
    return "Отчёт по визуализациям";
  }, [boardTurnHeaders]);

  async function handleExportBoard(format: BoardExportFormat): Promise<void> {
    const exportArtifacts = isOpenProjectMode
      ? format === "xlsx"
        ? openProjectTableArtifacts
        : openProjectReportArtifacts
      : visibleArtifacts;
    if (!exportArtifacts.length) {
      return;
    }
    setBoardExporting(format);
    setSourceError(null);
    try {
      const sections = isOpenProjectMode && format !== "xlsx"
        ? [
            { label: "Графики", artifact_ids: openProjectChartArtifacts.map((artifact) => artifact.id) },
            { label: "Таблицы", artifact_ids: openProjectTableArtifacts.map((artifact) => artifact.id) },
          ]
        : buildBoardExportSections(exportArtifacts, boardTurnHeaders);
      const title = isOpenProjectMode ? "Отчет OpenProject по проектам и списаниям" : boardExportTitle;
      await exportBoardReport(format, exportArtifacts, title, sections);
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBoardExporting(null);
    }
  }

  function compactBoardLayout(): void {
    setArtifactWidthMap((prev) => {
      const next = { ...prev };
      visibleArtifacts.forEach((artifact) => {
        next[artifact.id] = DEFAULT_ARTIFACT_WIDTH_UNITS;
      });
      return next;
    });
    setArtifactColStartMap({});
  }

  function handleRemoveFromBoard(artifactId: string): void {
    setArtifactOrderIds((prev) => prev.filter((id) => id !== artifactId));
    setArtifactHeightMap((prev) => {
      const next = { ...prev };
      delete next[artifactId];
      return next;
    });
    setArtifactWidthMap((prev) => {
      const next = { ...prev };
      delete next[artifactId];
      return next;
    });
    setArtifactColStartMap((prev) => {
      const next = { ...prev };
      delete next[artifactId];
      return next;
    });
    setMeasuredCardHeights((prev) => {
      const next = { ...prev };
      delete next[artifactId];
      return next;
    });
    boardCardRefs.current.delete(artifactId);
    onUnpinArtifact(artifactId);
  }

  function openRenameTurnHeader(header: TurnHeaderBoardLayout): void {
    setRenamingTurnHeader(header);
    setTurnTitleDraft(header.label);
  }

  function closeRenameTurnHeader(): void {
    setRenamingTurnHeader(null);
    setTurnTitleDraft("");
  }

  function saveTurnHeaderTitle(): void {
    if (!renamingTurnHeader) {
      return;
    }
    const nextTitle = turnTitleDraft.trim();
    setTurnTitleOverrides((prev) => {
      const next = { ...prev };
      if (nextTitle) {
        next[renamingTurnHeader.turnKey] = nextTitle;
      } else {
        delete next[renamingTurnHeader.turnKey];
      }
      return next;
    });
    closeRenameTurnHeader();
  }

  function resetTurnHeaderTitle(turnKey: string): void {
    setTurnTitleOverrides((prev) => {
      if (!prev[turnKey]) {
        return prev;
      }
      const next = { ...prev };
      delete next[turnKey];
      return next;
    });
  }

  function renderTurnHeader(header: TurnHeaderBoardLayout) {
    return (
      <div
        key={`turn-header-${header.turnKey}`}
        className="absolute left-0 z-[5] px-1"
        style={{
          top: header.topPx,
          width: `calc(100% - 2px)`,
          height: BOARD_TURN_HEADER_HEIGHT_PX,
        }}
      >
        <div className="flex h-full items-center justify-between gap-3 rounded-[20px] border-b border-border/50 bg-muted/30 px-3 py-1">
          <div className="min-w-0 whitespace-normal break-words text-[14px] font-semibold leading-tight text-foreground">
            {header.label}
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition hover:bg-background/70 hover:text-foreground"
                title="Действия блока"
              >
                <MoreHorizontal className="h-4 w-4" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => openRenameTurnHeader(header)}>
                <PenSquare className="mr-2 h-4 w-4" />
                Переименовать
              </DropdownMenuItem>
              {turnTitleOverrides[header.turnKey]?.trim() ? (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={() => resetTurnHeaderTitle(header.turnKey)}>
                    Сбросить название
                  </DropdownMenuItem>
                </>
              ) : null}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    );
  }

  function renderArtifactCard(artifact: ArtifactPayload, index: number) {
    const layout = artifactBoardLayouts.get(artifact.id);
    const contentHeight = artifactHeightMap[artifact.id] ?? estimateAutoHeight(artifact);
    const isDropTarget =
      draggedArtifactId &&
      draggedArtifactId !== artifact.id &&
      (dropIndex === index || dropIndex === index + 1);
    return (
      <motion.div
        key={artifact.id}
        layout="position"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        onDragOver={(event) => {
          event.preventDefault();
          const rect = event.currentTarget.getBoundingClientRect();
          const shouldInsertAfter = event.clientY > rect.top + rect.height / 2;
          setDropIndex(shouldInsertAfter ? index + 1 : index);
        }}
        onDrop={(event) => {
          event.preventDefault();
          handleDropAtIndex(dropIndex ?? index);
        }}
        className={`min-h-0 min-w-0 ${isDropTarget ? "ring-2 ring-primary/40 rounded-[28px]" : ""}`}
        ref={(node) => {
          if (node) {
            boardCardRefs.current.set(artifact.id, node);
          } else {
            boardCardRefs.current.delete(artifact.id);
          }
        }}
        style={
          layout
            ? {
                position: "absolute",
                top: layout.topPx,
                left: `calc(((100% - ${BOARD_GAP_PX * (GRID_COLUMNS - 1)}px) / ${GRID_COLUMNS}) * ${layout.colStart - 1} + ${BOARD_GAP_PX * (layout.colStart - 1)}px)`,
                width: `calc(((100% - ${BOARD_GAP_PX * (GRID_COLUMNS - 1)}px) / ${GRID_COLUMNS}) * ${layout.widthUnits} + ${BOARD_GAP_PX * (layout.widthUnits - 1)}px)`,
              }
            : undefined
        }
        data-artifact-id={artifact.id}
      >
          <div
            className={`group relative overflow-hidden rounded-[20px] border border-border/20 bg-card/55 transition ${
              draggedArtifactId === artifact.id ? "opacity-55" : "opacity-100"
            }`}
          >
            <div className="flex items-center justify-between gap-2 border-b border-border/10 bg-transparent px-3 py-2">
              <div
                draggable
                onDragStart={() => {
                  setDraggedArtifactId(artifact.id);
                  setDropIndex(index);
                }}
                onDragEnd={() => {
                  setDraggedArtifactId(null);
                  setDropIndex(null);
                }}
                className="inline-flex min-w-0 flex-1 cursor-grab items-center gap-1.5 px-1 py-0.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground active:cursor-grabbing"
                title="Перетащите для изменения порядка"
              >
                <GripVertical className="h-3.5 w-3.5 shrink-0" />
                <span className="min-w-0 normal-case tracking-normal">
                  <span className="block truncate font-semibold">
                    {artifact.text || artifact.type}
                  </span>
                  {artifact.type === "note" &&
                  typeof artifact.meta?.user_question === "string" &&
                  artifact.meta.user_question.trim() ? (
                    <span className="mt-0.5 block line-clamp-2 text-[10px] font-normal normal-case text-muted-foreground">
                      {artifact.meta.user_question}
                    </span>
                  ) : null}
                </span>
              </div>
              <button
                type="button"
                onClick={() => handleRemoveFromBoard(artifact.id)}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition hover:bg-rose-500/10 hover:text-rose-500"
                title="Удалить с доски"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="relative px-1 pb-1 pt-2">
              <ArtifactSurface
                artifact={artifact}
                variant="board"
                showCode={showCode}
                contentHeightPx={contentHeight}
              />

              <div
                onMouseDown={(event) => startPointerResize(artifact, "width-left", event)}
                onDoubleClick={() => {
                  setArtifactWidthMap((prev) => ({
                    ...prev,
                    [artifact.id]: DEFAULT_ARTIFACT_WIDTH_UNITS,
                  }));
                  setArtifactColStartMap((prev) => {
                    const next = { ...prev };
                    delete next[artifact.id];
                    return next;
                  });
                }}
                className="absolute left-0 top-12 bottom-10 z-10 w-2 cursor-ew-resize rounded-full bg-border/40 opacity-0 transition group-hover:opacity-100 hover:bg-primary/50"
                title="Ширина слева (двойной щелчок — сброс)"
              />
              <div
                onMouseDown={(event) => startPointerResize(artifact, "width", event)}
                onDoubleClick={() => {
                  setArtifactWidthMap((prev) => ({
                    ...prev,
                    [artifact.id]: DEFAULT_ARTIFACT_WIDTH_UNITS,
                  }));
                  setArtifactColStartMap((prev) => {
                    const next = { ...prev };
                    delete next[artifact.id];
                    return next;
                  });
                }}
                className="absolute right-0 top-12 bottom-10 z-10 w-2 cursor-ew-resize rounded-full bg-border/40 opacity-0 transition group-hover:opacity-100 hover:bg-primary/50"
                title="Ширина справа (двойной щелчок — сброс)"
              />
              <div
                onMouseDown={(event) => startPointerResize(artifact, "height", event)}
                onDoubleClick={() => {
                  const autoHeight = estimateAutoHeight(artifact);
                  setArtifactHeightMap((prev) => ({ ...prev, [artifact.id]: autoHeight }));
                }}
                className="absolute inset-x-3 bottom-1 z-10 h-2 cursor-ns-resize rounded-full bg-border/40 opacity-0 transition group-hover:opacity-100 hover:bg-primary/45"
                title="Высота (двойной щелчок — автоподбор)"
              />
              <div
                onMouseDown={(event) => startPointerResize(artifact, "both", event)}
                onDoubleClick={() => resetArtifactSize(artifact)}
                className="absolute bottom-0 right-0 z-20 h-4 w-4 cursor-nwse-resize rounded-tl-lg bg-primary/25 opacity-0 transition group-hover:opacity-80 hover:bg-primary/45"
                title="Ширина и высота (двойной щелчок — сброс обоих)"
              />
            </div>
          </div>
      </motion.div>
    );
  }

  function renderOpenProjectReport() {
    if (openProjectReportArtifacts.length === 0) {
      return (
        <div className="flex min-h-[420px] items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-10 text-center shadow-sm">
          <div className="max-w-[520px]">
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-3xl bg-cyan-500/10 text-cyan-500">
              <PlugZap className="h-8 w-8" />
            </div>
            <h3 className="mb-3 text-2xl font-bold tracking-tight">Отчет OpenProject пока не сформирован</h3>
            <p className="text-[15px] leading-relaxed text-muted-foreground">
              Откройте вкладку источников и запустите синхронизацию OpenProject.
            </p>
          </div>
        </div>
      );
    }

    const syncedAt = openProjectReportArtifacts[0]?.meta?.synced_at;
    return (
      <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto pr-2">
        <div className="space-y-5 pb-3">
          <div className="rounded-[28px] border border-border/50 bg-card/45 p-5 shadow-sm">
            <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
              <div>
                <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-cyan-500">
                  OpenProject
                </div>
                <h2 className="mt-1 text-2xl font-bold tracking-tight">Отчет по проектам и списаниям</h2>
              </div>
              {typeof syncedAt === "string" ? (
                <div className="text-sm text-muted-foreground">
                  Обновлено: {new Date(syncedAt).toLocaleString("ru-RU")}
                </div>
              ) : null}
            </div>
          </div>

          {openProjectChartArtifacts.length > 0 ? (
            <div className="grid gap-4 xl:grid-cols-2">
              {openProjectChartArtifacts.map((artifact) => (
                <div key={artifact.id} className="min-h-[360px] rounded-[24px] border border-border/50 bg-card/60 p-4 shadow-sm">
                  <h3 className="mb-3 text-base font-bold tracking-tight">{artifact.text || "График"}</h3>
                  <ArtifactSurface artifact={artifact} showCode={false} contentHeightPx={300} variant="board" />
                </div>
              ))}
            </div>
          ) : null}

          {openProjectTableArtifacts.map((artifact) => (
            <div key={artifact.id} className="rounded-[24px] border border-border/50 bg-card/60 p-4 shadow-sm">
              <h3 className="mb-3 text-base font-bold tracking-tight">{artifact.text || "Таблица"}</h3>
              <ArtifactSurface artifact={artifact} showCode={false} contentHeightPx={360} variant="board" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const isEditing = Boolean(editingConnection);

  return (
    <div className="flex h-full min-h-0 flex-col space-y-6">
      <div className="flex min-h-12 items-center justify-between gap-3">
        <div className="inline-flex h-12 items-center rounded-[18px] border border-border/50 bg-secondary/40 p-1">
          <button
            type="button"
            onClick={() => setActiveTab("visualizations")}
            className={`inline-flex h-10 items-center gap-2 rounded-[14px] px-4 text-sm font-bold transition-all ${
              activeTab === "visualizations"
                ? "bg-card text-foreground shadow-sm ring-1 ring-border/20"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Grid2X2 className="h-4 w-4" />
            Визуализации
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("sources")}
            className={`inline-flex h-10 items-center gap-2 rounded-[14px] px-4 text-sm font-bold transition-all ${
              activeTab === "sources"
                ? "bg-card text-foreground shadow-sm ring-1 ring-white"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Upload className="h-4 w-4" />
            Источники
          </button>
        </div>

        <div className="flex h-12 shrink-0 items-center gap-2">
          {activeTab === "sources" ? (
            <button
              type="button"
              onClick={() => {
                if (isOpenProjectMode) {
                  return;
                }
                resetForm(null);
                setDialogOpen(true);
              }}
              disabled={isOpenProjectMode}
              className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary/60 px-4 text-sm font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Database className="h-4 w-4" />
              Подключение
            </button>
          ) : null}
          {activeTab === "visualizations" &&
          ((isOpenProjectMode && openProjectReportArtifacts.length > 0) ||
            (!isOpenProjectMode && visibleArtifacts.length > 0)) ? (
            <>
              <button
                type="button"
                onClick={() => void handleExportBoard("xlsx")}
                disabled={boardExporting !== null}
                className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary/60 px-4 text-sm font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                title="Скачать отчет как Excel"
              >
                {boardExporting === "xlsx" ? (
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                ) : (
                  <FileDown className="h-4 w-4" />
                )}
                Excel
              </button>
              {!isOpenProjectMode ? (
                <button
                  type="button"
                  onClick={() => void handleExportBoard("docx")}
                  disabled={boardExporting !== null}
                  className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary/60 px-4 text-sm font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                  title="Скачать доску как Word-отчёт"
                >
                  {boardExporting === "docx" ? (
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                  ) : (
                    <FileDown className="h-4 w-4" />
                  )}
                  DOCX
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => void handleExportBoard("pdf")}
                disabled={boardExporting !== null}
                className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary/60 px-4 text-sm font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                title="Скачать доску как PDF-отчёт"
              >
                {boardExporting === "pdf" ? (
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                ) : (
                  <FileDown className="h-4 w-4" />
                )}
                PDF
              </button>
            </>
          ) : null}
          {activeTab === "visualizations" && visibleArtifacts.length >= 2 ? (
            <button
              type="button"
              onClick={compactBoardLayout}
              className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary/60 px-4 text-sm font-bold text-muted-foreground transition-all hover:bg-muted hover:text-foreground"
              title="Сбросить ширину всех карточек до половины доски"
            >
              <RefreshCcw className="h-4 w-4" />
              Уплотнить
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex h-12 w-12 items-center justify-center rounded-2xl border border-border/50 bg-secondary/60 text-muted-foreground transition-all hover:bg-muted hover:text-foreground"
          >
            <Plus className="h-5 w-5" />
          </button>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        className="hidden"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length > 0) {
            void handleFilePick(files);
          }
          event.currentTarget.value = "";
        }}
      />

      <input
        ref={ragInputRef}
        type="file"
        accept=".txt,.md,.markdown,.pdf,.docx,.html,.htm,.csv,.json,text/plain,text/markdown,application/pdf"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            void handleRagFilePick(file);
          }
          event.currentTarget.value = "";
        }}
      />

      <div className="min-h-0 flex-1">
        <AnimatePresence mode="wait">
          {activeTab === "visualizations" ? (
            <motion.div
              key="visualizations"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex h-full min-h-0 flex-col"
            >
              {sourceError && activeTab === "visualizations" ? (
                <div className="mb-3 rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                  {sourceError}
                </div>
              ) : null}
              {isOpenProjectMode ? (
                renderOpenProjectReport()
              ) : visibleArtifacts.length === 0 ? (
                <div className="flex min-h-[420px] items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-10 text-center shadow-sm">
                  <div className="max-w-[520px]">
                    <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-3xl bg-primary/10 text-primary">
                      <Grid2X2 className="h-8 w-8" />
                    </div>
                    <h3 className="mb-3 text-2xl font-bold tracking-tight">Пока пусто</h3>
                    <p className="text-[15px] leading-relaxed text-muted-foreground">
                      Добавляйте артефакты в дашборд кнопкой в чате рядом с каждым артефактом.
                    </p>
                  </div>
                </div>
              ) : (
                <div
                  ref={boardScrollRef}
                  className="custom-scrollbar min-h-0 flex-1 overflow-y-auto pr-2"
                >
                  <div
                    className="relative pb-2"
                    style={{ minHeight: Math.max(artifactBoardHeight, 240) }}
                  >
                    {turnHeaderLayouts.map((header) => renderTurnHeader(header))}
                    {visibleArtifacts.map((artifact, index) =>
                      renderArtifactCard(artifact, index),
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="sources"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="custom-scrollbar flex h-full min-h-0 flex-col space-y-6 overflow-y-auto pr-2"
            >
              <div className="grid gap-4 xl:grid-cols-4">
                <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-7 text-center shadow-sm">
                  <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-primary/10 text-primary">
                    <Upload className="h-7 w-7" />
                  </div>
                  <h3 className="text-xl font-bold tracking-tight">CSV/XLSX</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Загрузите CSV/XLSX-файлы и используйте их как источник данных.
                  </p>
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    disabled={isOpenProjectMode}
                    className="mt-6 inline-flex items-center justify-center gap-2 rounded-2xl bg-primary px-5 py-3 font-bold text-primary-foreground shadow-lg shadow-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Upload className="h-4 w-4" />
                    Загрузить CSV/XLSX
                  </button>
                </div>

                <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-7 text-center shadow-sm">
                  <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-sky-500/10 text-sky-500">
                    <Database className="h-7 w-7" />
                  </div>
                  <h3 className="text-xl font-bold tracking-tight">База данных</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Подключите PostgreSQL или ClickHouse и выберите сохраненное соединение.
                  </p>
                  <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => void handleOpenSavedConnections()}
                      disabled={isOpenProjectMode}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-sky-500 px-5 py-3 font-bold text-sky-950 shadow-lg shadow-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Database className="h-4 w-4" />
                      Выбрать подключение
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (isOpenProjectMode) {
                          return;
                        }
                        resetForm(null);
                        setDialogOpen(true);
                      }}
                      disabled={isOpenProjectMode}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-border/50 bg-card/70 px-5 py-3 font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Plus className="h-4 w-4" />
                      Новое
                    </button>
                  </div>
                </div>

                <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-7 text-center shadow-sm">
                  <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-cyan-500/10 text-cyan-500">
                    <PlugZap className="h-7 w-7" />
                  </div>
                  <h3 className="text-xl font-bold tracking-tight">OpenProject</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Выгрузите проекты, задачи, списания и участников в PostgreSQL.
                  </p>
                  <button
                    type="button"
                    onClick={() => setSourceSection("openproject")}
                    className="mt-6 inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-500 px-5 py-3 font-bold text-cyan-950 shadow-lg shadow-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    <PlugZap className="h-4 w-4" />
                    OpenProject
                  </button>
                </div>

                <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-7 text-center shadow-sm">
                  <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-emerald-500/10 text-emerald-500">
                    <BookOpen className="h-7 w-7" />
                  </div>
                  <h3 className="text-xl font-bold tracking-tight">База знаний</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Загрузите документы в LightRAG и используйте их как источник знаний.
                  </p>
                  <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => ragInputRef.current?.click()}
                      disabled={ragUploading || isOpenProjectMode}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-5 py-3 font-bold text-emerald-950 shadow-lg shadow-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {ragUploading ? (
                        <LoaderCircle className="h-4 w-4 animate-spin" />
                      ) : (
                        <FileUp className="h-4 w-4" />
                      )}
                      Загрузить документы
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleOpenKnowledgeBase()}
                      disabled={isOpenProjectMode}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-border/50 bg-card/70 px-5 py-3 font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <BookOpen className="h-4 w-4" />
                      Открыть
                    </button>
                  </div>
                </div>
              </div>

              <div className="rounded-[22px] border border-border/50 bg-secondary/20 px-5 py-4">
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div className="min-w-0">
                    <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                      Текущий источник
                    </div>
                    <div className="mt-1 truncate text-base font-bold text-foreground">
                      {activeSourceText?.label ?? "Источник не выбран"}
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      {activeSourceText ? `${activeSourceText.meta} · ${activeSourceText.status}` : "Источник данных пока не выбран"}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={handleSwitchSource}
                      className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted"
                    >
                      Сменить
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleClearSource()}
                      disabled={clearingSource || !activeSource.source_type || isOpenProjectMode}
                      className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-muted-foreground transition-all hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {clearingSource ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
                      Очистить
                    </button>
                  </div>
                </div>
              </div>

              <div className="inline-flex w-fit items-center rounded-[18px] border border-border/50 bg-secondary/40 p-1">
                <button
                  type="button"
                  onClick={() => {
                    if (!isOpenProjectMode) {
                      setSourceSection("db");
                    }
                  }}
                  disabled={isOpenProjectMode}
                  className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
                    sourceSection === "db"
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20 dark:ring-white"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Database className="h-4 w-4" />
                  Подключения к БД
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (!isOpenProjectMode) {
                      setSourceSection("csv");
                    }
                  }}
                  disabled={isOpenProjectMode}
                  className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
                    sourceSection === "csv"
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20 dark:ring-white"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <FileText className="h-4 w-4" />
                  Загрузки CSV/XLSX
                </button>
                <button
                  type="button"
                  onClick={() => setSourceSection("openproject")}
                  className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
                    sourceSection === "openproject"
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20 dark:ring-white"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <PlugZap className="h-4 w-4" />
                  OpenProject
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (isOpenProjectMode) {
                      return;
                    }
                    setSourceSection("rag");
                    void loadRagDocuments();
                  }}
                  disabled={isOpenProjectMode}
                  className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
                    sourceSection === "rag"
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20 dark:ring-white"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <BookOpen className="h-4 w-4" />
                  База знаний
                </button>
              </div>

              {sourceSection === "db" ? (
              <div ref={dbListRef} className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
                  <div className="mb-5 flex items-center justify-between">
                    <div>
                      <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                        DB Connections
                      </div>
                      <h3 className="mt-2 text-xl font-bold tracking-tight">Подключения к БД</h3>
                    </div>
                    <button
                      type="button"
                      onClick={() => void loadConnections()}
                      className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-border/50 bg-secondary/70 text-muted-foreground transition-all hover:text-foreground"
                    >
                      <RefreshCcw className={`h-4 w-4 ${connectionsLoading ? "animate-spin" : ""}`} />
                    </button>
                  </div>

                  <div className="space-y-3">
                    {connectionsLoading ? (
                      <div className="flex items-center gap-3 rounded-[22px] border border-border/50 bg-secondary/30 px-4 py-4 text-sm text-muted-foreground">
                        <LoaderCircle className="h-4 w-4 animate-spin" />
                        Загружаю подключения...
                      </div>
                    ) : (
                      visibleConnections.map((connection) => {
                        const isActive =
                          activeSource.source_type === "db_connection" &&
                          activeSource.source_ref_id === connection.id;
                        const testedOk = connection.last_test_ok === true;
                        const hasSchema = typeof connection.options_json?.schema === "string" && connection.options_json.schema.trim().length > 0;
                        const canUse = testedOk && hasSchema;
                        return (
                          <div key={connection.id} className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-4">
                            {/* Header: info + three-dots menu */}
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="text-sm font-bold text-foreground">{connection.name}</span>
                                  <span className="rounded-full border border-border/50 bg-card/70 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                                    {connection.db_type}
                                  </span>
                                  {isActive ? <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-emerald-400">активно</span> : null}
                                </div>
                                <div className="mt-1 truncate text-xs text-muted-foreground">
                                  {connection.host}{connection.port ? `:${connection.port}` : ""}{connection.database ? ` / ${connection.database}` : ""}
                                </div>
                                <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                                  <span className="inline-flex items-center gap-1">
                                    <ShieldCheck className="h-3 w-3" />
                                    {connection.password_present ? "секрет сохранен" : "секрет не задан"}
                                  </span>
                                  {testedOk ? (
                                    <span className="inline-flex items-center gap-1 text-emerald-500">
                                      <CheckCircle2 className="h-3 w-3" />
                                      connection ok
                                    </span>
                                  ) : null}
                                </div>
                              </div>

                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <button type="button" className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl border border-border/50 bg-card/70 text-muted-foreground transition-all hover:text-foreground">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" className="min-w-[150px]">
                                  <DropdownMenuItem
                                    onClick={() => void handleTestConnection(connection)}
                                    disabled={testingId === connection.id}
                                    className="gap-2 text-xs"
                                  >
                                    {testingId === connection.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                                    Тест
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => { resetForm(connection); setDialogOpen(true); }}
                                    className="gap-2 text-xs"
                                  >
                                    <PenSquare className="h-3.5 w-3.5" />
                                    Изменить
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    onClick={() => void handleDeleteConnection(connection)}
                                    disabled={deletingId === connection.id}
                                    className="gap-2 text-xs text-rose-400 focus:text-rose-400"
                                  >
                                    {deletingId === connection.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                    Удалить
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>

                            {/* Action row: Use button + schema select */}
                            <div className="mt-3 flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => void handleBindConnection(connection)}
                                disabled={!sessionId || bindingId === connection.id || !canUse}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                {bindingId === connection.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}
                                Использовать
                              </button>

                              <div className="ml-auto flex items-center gap-1.5">
                                {testedOk ? (
                                  <>
                                    <button
                                      type="button"
                                      onClick={() => void loadSchemasForConnection(connection.id)}
                                      disabled={schemasLoadingId === connection.id}
                                      className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl border border-border/50 bg-card/60 text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                      <RefreshCcw className={`h-3.5 w-3.5 ${schemasLoadingId === connection.id ? "animate-spin" : ""}`} />
                                    </button>
                                    <Select
                                      value={typeof connection.options_json?.schema === "string" ? connection.options_json.schema : ""}
                                      onValueChange={(value) => void handleSelectSchema(connection, value)}
                                      disabled={schemasLoadingId === connection.id}
                                    >
                                      <SelectTrigger className="h-8 w-44 rounded-xl border border-border/50 bg-card/60 px-3 text-xs text-muted-foreground">
                                        <SelectValue placeholder="Выберите схему" />
                                      </SelectTrigger>
                                      <SelectContent>
                                        {!connectionSchemas[connection.id] && typeof connection.options_json?.schema === "string" && connection.options_json.schema ? (
                                          <SelectItem value={connection.options_json.schema}>{connection.options_json.schema}</SelectItem>
                                        ) : null}
                                        {(connectionSchemas[connection.id] ?? []).map((s) => (
                                          <SelectItem key={s.name} value={s.name}>{s.display_name}</SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  </>
                                ) : (
                                  <span className="text-[11px] text-muted-foreground/60">Сначала запустите тест</span>
                                )}
                              </div>
                            </div>

                            {connection.last_error ? (
                              <div className="mt-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-200">
                                {connection.last_error}
                              </div>
                            ) : null}
                          </div>
                        );
                      })
                    )}
                    {!connectionsLoading && visibleConnections.length === 0 ? (
                      <div className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
                        Пока нет сохраненных подключений к БД.
                      </div>
                    ) : null}
                  </div>
              </div>
              ) : sourceSection === "openproject" ? (
              <div className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
                <div className="mb-5 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                      OpenProject
                    </div>
                    <h3 className="mt-2 text-xl font-bold tracking-tight">Синхронизация OpenProject</h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => void handleBindOpenProjectSource()}
                    disabled={!sessionId || bindingOpenProject}
                    className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {bindingOpenProject ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}
                    {bindingOpenProject ? "Синхронизация..." : "Синхронизировать"}
                  </button>
                </div>

                <div className="mb-4 grid gap-3 md:grid-cols-2">
                  <label className="space-y-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">URL OpenProject</span>
                    <input
                      value={openProjectForm.baseUrl}
                      onChange={(event) => setOpenProjectForm((prev) => ({ ...prev, baseUrl: event.target.value }))}
                      className="h-11 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40"
                      placeholder="http://localhost:8080"
                    />
                  </label>
                  <label className="space-y-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">API-ключ</span>
                    <input
                      value={openProjectForm.apiKey}
                      onChange={(event) => setOpenProjectForm((prev) => ({ ...prev, apiKey: event.target.value }))}
                      type="password"
                      className="h-11 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40"
                      placeholder="Если пусто, используется значение из .env"
                    />
                  </label>
                  <label className="space-y-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Проект</span>
                    <div className="flex gap-2">
                      <Select
                        value={openProjectForm.project || "__env__"}
                        onValueChange={(value) =>
                          setOpenProjectForm((prev) => ({ ...prev, project: value === "__env__" ? "" : value }))
                        }
                      >
                        <SelectTrigger className="h-11 flex-1 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                          <SelectValue placeholder="Выберите проект" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__env__">Все проекты</SelectItem>
                          {openProjectProjects.map((project) => (
                            <SelectItem key={project.id || project.identifier} value={project.identifier || project.id}>
                              {project.name} ({project.identifier || project.id})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <button
                        type="button"
                        onClick={() => void handleLoadOpenProjectProjects()}
                        disabled={!sessionId || openProjectProjectsLoading}
                        className="inline-flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl border border-border/60 bg-secondary/40 text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                        title="Загрузить проекты"
                      >
                        <RefreshCcw className={`h-4 w-4 ${openProjectProjectsLoading ? "animate-spin" : ""}`} />
                      </button>
                    </div>
                  </label>
                  <label className="space-y-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Период, дней</span>
                    <input
                      value={openProjectForm.days}
                      onChange={(event) => setOpenProjectForm((prev) => ({ ...prev, days: event.target.value }))}
                      inputMode="numeric"
                      className="h-11 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40"
                      placeholder="90"
                    />
                  </label>
                </div>

                <div className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-4">
                  <div className="text-sm font-bold text-foreground">
                    {activeSource.source_type === "openproject" ? "OpenProject активен" : "Готово к синхронизации"}
                  </div>
                  <div className="mt-1 text-sm text-muted-foreground">
                    Таблицы пересоздаются в PostgreSQL и сразу попадают в дашборд аналитика.
                  </div>
                  {openProjectSync ? (
                    <div className="mt-4 grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                      {Object.entries(openProjectSync.tables).map(([table, rows]) => (
                        <div key={table} className="rounded-2xl border border-border/50 bg-card/50 px-3 py-3">
                          <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{table}</div>
                          <div className="mt-1 text-lg font-bold text-foreground">{rows.toLocaleString("ru-RU")}</div>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {openProjectSync ? (
                    <div className="mt-3 text-xs text-muted-foreground">
                      Схема: {openProjectSync.schema_name} · синхронизировано: {new Date(openProjectSync.synced_at).toLocaleString("ru-RU")}
                    </div>
                  ) : null}
                </div>
              </div>
              ) : sourceSection === "rag" ? (
              <div ref={ragListRef} className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
                <div className="mb-5 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                      LightRAG
                    </div>
                    <h3 className="mt-2 text-xl font-bold tracking-tight">База знаний</h3>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => ragInputRef.current?.click()}
                      disabled={ragUploading}
                      className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {ragUploading ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <FileUp className="h-3.5 w-3.5" />}
                      Загрузить документы
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleBindRagSource()}
                      disabled={!sessionId || bindingRag || activeSource.source_type === "rag"}
                      className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {bindingRag ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <BookOpen className="h-3.5 w-3.5" />}
                      {activeSource.source_type === "rag" ? "Уже используется" : "Использовать базу знаний"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void loadRagDocuments()}
                      disabled={ragDocumentsLoading}
                      className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-border/50 bg-secondary/70 text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <RefreshCcw className={`h-4 w-4 ${ragDocumentsLoading ? "animate-spin" : ""}`} />
                    </button>
                  </div>
                </div>

                <div className="space-y-3">
                  {ragDocumentsLoading && ragDocuments.length === 0 ? (
                    <div className="flex items-center gap-3 rounded-[22px] border border-border/50 bg-secondary/30 px-4 py-4 text-sm text-muted-foreground">
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      Загружаю документы...
                    </div>
                  ) : null}

                  {ragDocuments.map((document, index) => {
                    const status = ragStatusLabel(document.status);
                    const isProcessing = isRagProcessing(document.status);
                    const isFailed = status === "failed";
                    const isDeletingDocument = Boolean(document.id && deletingRagDocumentId === document.id);
                    return (
                      <div key={document.id || document.file_path || index} className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-4">
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="truncate text-sm font-bold text-foreground">
                                {document.file_path || document.id || "Документ"}
                              </span>
                              <span
                                className={`rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                                  isFailed
                                    ? "border-rose-500/20 bg-rose-500/10 text-rose-400"
                                    : status === "processed"
                                      ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                                      : "border-amber-500/20 bg-amber-500/10 text-amber-500"
                                }`}
                              >
                                {formatRagStatusLabel(document.status)}
                              </span>
                            </div>
                            <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                              {document.chunks_count != null ? <span>{document.chunks_count} фрагментов</span> : null}
                              {document.content_length != null ? <span>{document.content_length} байт</span> : null}
                              {document.updated_at ? <span>{new Date(document.updated_at).toLocaleString("ru-RU")}</span> : null}
                            </div>
                            {document.error_msg ? (
                              <div className="mt-3 rounded-2xl border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-700 dark:text-rose-200">
                                {document.error_msg}
                              </div>
                            ) : null}
                          </div>
                          <div className="flex shrink-0 items-center gap-2">
                            {isProcessing ? <LoaderCircle className="h-4 w-4 animate-spin text-muted-foreground" /> : null}
                            <button
                              type="button"
                              onClick={() => void handleDeleteRagDocument(document)}
                              disabled={!document.id || isDeletingDocument}
                              title="Удалить документ"
                              aria-label="Удалить документ"
                              className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-border/50 bg-card/70 text-muted-foreground transition-all hover:border-rose-500/30 hover:text-rose-500 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              {isDeletingDocument ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {!ragDocumentsLoading && ragDocuments.length === 0 ? (
                    <div className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
                      В базе знаний пока нет документов.
                    </div>
                  ) : null}
                </div>
              </div>
              ) : (
              <div className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
                <div className="mb-4 text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                  Последние загрузки
                </div>
                {hasDataset ? (
                  <div className="mb-4 rounded-[24px] border border-border/50 bg-secondary/20 px-5 py-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div>
                        <div className="text-[15px] font-bold text-foreground">
                          {datasetName || "CSV dataset"}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {activeSource.source_type === "csv"
                            ? "CSV уже выбран как активный источник"
                            : "Файл уже загружен в сессию и доступен для повторного выбора"}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => void handleBindCsvSource()}
                        disabled={
                          bindingCsv ||
                          !sessionId ||
                          !datasetName ||
                          activeSource.source_type === "csv"
                        }
                        className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {bindingCsv ? (
                          <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <FileText className="h-3.5 w-3.5" />
                        )}
                        Использовать CSV/XLSX
                      </button>
                    </div>
                    {csvSources.length > 0 ? (
                      <div className="mt-4 border-t border-border/40 pt-3">
                        <div className="space-y-2">
                          {csvSources.map((source) => (
                            <div
                              key={source.alias}
                              className="flex flex-col gap-1 text-sm md:flex-row md:items-center md:justify-between"
                            >
                              <span className="min-w-0 truncate font-semibold text-foreground">
                                {source.display_name || source.file_name || source.alias}
                              </span>
                              {source.csv_table_names?.length ? (
                                <span className="min-w-0 truncate text-xs text-muted-foreground">
                                  DuckDB: {source.csv_table_names.join(", ")}
                                </span>
                              ) : null}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
                <div className="space-y-4">
                  {recentUploads.length > 0 ? (
                    recentUploads.map((item) => (
                      <div key={`${item.name}-${item.date}`} className="flex items-center justify-between rounded-[24px] border border-border/50 bg-card/50 px-5 py-5 shadow-sm">
                        <div className="flex items-center gap-4">
                          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary/70 text-muted-foreground">
                            <FileText className="h-5 w-5" />
                          </div>
                          <div>
                            <div className="text-[15px] font-bold">{item.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {item.tables.length > 0 ? `${item.date} · ${item.tables.join(", ")}` : item.date}
                            </div>
                          </div>
                        </div>
                        <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-emerald-400">
                          analyzed
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-[24px] border border-border/50 bg-card/40 px-5 py-6 text-sm text-muted-foreground">
                      Пока нет загруженных CSV/XLSX файлов.
                    </div>
                  )}
                </div>
              </div>
              )}

              {sourceError ? (
                <div className="rounded-[24px] border border-rose-500/20 bg-rose-500/10 px-5 py-4 text-sm text-rose-700 dark:text-rose-200">
                  {sourceError}
                </div>
              ) : null}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <Dialog
        open={Boolean(renamingTurnHeader)}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) {
            closeRenameTurnHeader();
          }
        }}
      >
        <DialogContent className="max-w-lg rounded-[28px] border-border/60 bg-background/98 p-0 backdrop-blur-xl">
          <div className="p-7">
            <DialogHeader className="mb-5 text-left">
              <DialogTitle className="text-xl font-bold tracking-tight">
                Переименовать блок
              </DialogTitle>
              <DialogDescription className="text-sm leading-relaxed">
                {renamingTurnHeader?.label ?? ""}
              </DialogDescription>
            </DialogHeader>
            <input
              autoFocus
              value={turnTitleDraft}
              onChange={(event) => setTurnTitleDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  saveTurnHeaderTitle();
                }
              }}
              className="h-11 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm text-foreground outline-none transition focus:border-primary/40"
            />
            <DialogFooter className="mt-6 flex-col-reverse gap-2 sm:flex-row">
              <button
                type="button"
                onClick={closeRenameTurnHeader}
                className="inline-flex h-10 items-center justify-center rounded-xl border border-border/60 bg-card/70 px-4 text-sm font-bold text-muted-foreground transition-all hover:bg-muted hover:text-foreground"
              >
                Отмена
              </button>
              <button
                type="button"
                onClick={saveTurnHeaderTitle}
                className="inline-flex h-10 items-center justify-center rounded-xl bg-primary px-4 text-sm font-bold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:opacity-90"
              >
                Сохранить
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={uploadDialogOpen}
        onOpenChange={(nextOpen) => {
          setUploadDialogOpen(nextOpen);
          if (!nextOpen) {
            resetUploadDialog();
          }
        }}
      >
        <DialogContent className="max-w-2xl rounded-[28px] border-border/60 bg-background/98 p-0 backdrop-blur-xl">
          <div className="p-7">
            <DialogHeader className="mb-6 text-left">
              <DialogTitle className="flex items-center gap-2 text-2xl font-bold tracking-tight">
                <SlidersHorizontal className="h-5 w-5 text-muted-foreground" />
                Настройки загрузки
              </DialogTitle>
              <DialogDescription className="text-sm leading-relaxed">
                {pendingUploadFiles.length > 0
                  ? pendingUploadFiles.map((file) => file.name).join(", ")
                  : "Файлы не выбраны"}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-5">
              <div className="grid gap-2 sm:grid-cols-2">
                {renderPreprocessingCheckbox("enabled", "Применять предобработку")}
                {renderPreprocessingCheckbox("detect_csv_separator", "Авторазделитель CSV")}
                {renderPreprocessingCheckbox("detect_header_row", "Искать строку заголовков XLSX")}
                {renderPreprocessingCheckbox("normalize_empty_values", "Нормализовать пустые значения")}
                {renderPreprocessingCheckbox("drop_empty_rows", "Удалять пустые строки")}
                {renderPreprocessingCheckbox("drop_empty_columns", "Удалять пустые колонки")}
                {renderPreprocessingCheckbox("drop_sparse_rows", "Удалять разреженные строки")}
                {renderPreprocessingCheckbox("unique_column_names", "Уникальные имена колонок")}
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-2 text-sm">
                  <span className="font-semibold text-foreground">Строк для поиска заголовка</span>
                  <input
                    type="number"
                    min={1}
                    max={500}
                    step={1}
                    disabled={!preprocessingOptions.enabled}
                    value={preprocessingOptions.header_scan_rows}
                    onChange={(event) => {
                      const next = Math.max(1, Math.min(500, Number(event.target.value) || 1));
                      updatePreprocessingOption("header_scan_rows", next);
                    }}
                    className="h-10 w-full rounded-xl border border-border/60 bg-card/70 px-3 text-sm text-foreground outline-none focus:border-primary disabled:cursor-not-allowed disabled:opacity-50"
                  />
                </label>
                <label className="space-y-2 text-sm">
                  <span className="font-semibold text-foreground">Порог разреженности</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    disabled={!preprocessingOptions.enabled || !preprocessingOptions.drop_sparse_rows}
                    value={preprocessingOptions.sparse_row_min_ratio}
                    onChange={(event) => {
                      const next = Math.max(0, Math.min(1, Number(event.target.value) || 0));
                      updatePreprocessingOption("sparse_row_min_ratio", next);
                    }}
                    className="h-10 w-full rounded-xl border border-border/60 bg-card/70 px-3 text-sm text-foreground outline-none focus:border-primary disabled:cursor-not-allowed disabled:opacity-50"
                  />
                </label>
              </div>
            </div>

            <DialogFooter className="mt-7 flex-col-reverse gap-2 sm:flex-row">
              <button
                type="button"
                onClick={() => setUploadDialogOpen(false)}
                className="inline-flex h-10 items-center justify-center rounded-xl border border-border/60 bg-card/70 px-4 text-sm font-bold text-muted-foreground transition-all hover:bg-muted hover:text-foreground"
              >
                Отмена
              </button>
              <button
                type="button"
                onClick={() => void handleConfirmUpload()}
                disabled={pendingUploadFiles.length === 0}
                className="inline-flex h-10 items-center justify-center gap-2 rounded-xl bg-primary px-4 text-sm font-bold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Upload className="h-4 w-4" />
                Загрузить
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={dialogOpen} onOpenChange={(nextOpen) => {
        setDialogOpen(nextOpen);
        if (!nextOpen) {
          resetForm(null);
        }
      }}>
        <DialogContent className="max-w-2xl rounded-[28px] border-border/60 bg-background/98 p-0 backdrop-blur-xl">
          <div className="p-7">
            <DialogHeader className="mb-6 text-left">
              <DialogTitle className="text-2xl font-bold tracking-tight">
                {isEditing ? "Изменить подключение" : "Новое подключение"}
              </DialogTitle>
              <DialogDescription className="text-sm leading-relaxed">
                Секрет хранится только на бэкенде. После сохранения пароль больше не возвращается в интерфейс.
              </DialogDescription>
            </DialogHeader>

            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Название</span>
                <input value={form.name} onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="Хранилище продаж" />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Тип БД</span>
                <Select value={form.dbType} onValueChange={(value) => setForm((prev) => ({ ...prev, dbType: value as DBConnectionType, port: defaultPortFor(value as DBConnectionType) }))}>
                  <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="postgresql">PostgreSQL</SelectItem>
                    <SelectItem value="clickhouse">ClickHouse</SelectItem>
                  </SelectContent>
                </Select>
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Хост</span>
                <input value={form.host} onChange={(event) => setForm((prev) => ({ ...prev, host: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="db.example.com" />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Порт</span>
                <input value={form.port} onChange={(event) => setForm((prev) => ({ ...prev, port: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder={defaultPortFor(form.dbType)} />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">База данных</span>
                <input value={form.database} onChange={(event) => setForm((prev) => ({ ...prev, database: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder={form.dbType === "clickhouse" ? "default" : "analytics"} />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Пользователь</span>
                <input value={form.username} onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="readonly_user" />
              </label>

              {form.dbType === "postgresql" ? (
                <label className="space-y-2">
                  <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Режим SSL</span>
                  <Select value={form.sslmode} onValueChange={(value) => setForm((prev) => ({ ...prev, sslmode: value }))}>
                    <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="disable">отключить</SelectItem>
                      <SelectItem value="prefer">предпочесть</SelectItem>
                      <SelectItem value="require">требовать</SelectItem>
                    </SelectContent>
                  </Select>
                </label>
              ) : (
                <label className="space-y-2">
                  <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Транспорт</span>
                  <Select value={form.secure ? "https" : "http"} onValueChange={(value) => setForm((prev) => ({ ...prev, secure: value === "https" }))}>
                    <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="http">HTTP</SelectItem>
                      <SelectItem value="https">HTTPS</SelectItem>
                    </SelectContent>
                  </Select>
                </label>
              )}

              <div className="space-y-2 md:col-span-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Секрет</span>
                {isEditing ? (
                  <div className="grid gap-3 md:grid-cols-[220px_1fr]">
                    <Select value={form.secretMode} onValueChange={(value) => setForm((prev) => ({ ...prev, secretMode: value as SecretMode, password: value === "replace" ? prev.password : "" }))}>
                      <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="keep">Оставить как есть</SelectItem>
                        <SelectItem value="replace">Заменить секрет</SelectItem>
                        <SelectItem value="clear">Очистить секрет</SelectItem>
                      </SelectContent>
                    </Select>
                    <input value={form.password} onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value, secretMode: prev.secretMode === "keep" ? "replace" : prev.secretMode }))} disabled={form.secretMode !== "replace"} type="password" className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40 disabled:cursor-not-allowed disabled:opacity-60" placeholder={editingConnection?.password_present ? "Секрет сохранен на бэкенде" : "Введите пароль"} />
                  </div>
                ) : (
                  <input value={form.password} onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))} type="password" className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="Введите пароль или токен" />
                )}
              </div>
            </div>

            {sourceError ? (
              <div className="mt-5 rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                {sourceError}
              </div>
            ) : null}

            <DialogFooter className="mt-6">
              <button type="button" onClick={() => { setDialogOpen(false); resetForm(null); }} className="inline-flex h-11 items-center justify-center rounded-2xl border border-border/50 px-5 text-sm font-bold text-muted-foreground transition-all hover:text-foreground">
                Отмена
              </button>
              <button type="button" disabled={submitting || !form.name.trim() || !form.host.trim()} onClick={() => void handleSubmitConnection()} className="inline-flex h-11 items-center justify-center gap-2 rounded-2xl bg-primary px-5 text-sm font-bold text-primary-foreground shadow-lg shadow-primary/20 disabled:cursor-not-allowed disabled:opacity-60">
                {submitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Database className="h-4 w-4" />}
                {isEditing ? "Сохранить изменения" : "Создать подключение"}
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

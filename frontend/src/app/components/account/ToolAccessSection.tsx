import { forwardRef, useEffect, useImperativeHandle, useMemo, useState } from "react";
import { AlertCircle, Download, FilePenLine, Plus, PlugZap, Server, Sparkles, Trash2 } from "lucide-react";
import {
  deleteAdminMcpServer,
  exportSkillsArchive,
  getAdminSkillDetail,
  getUserTools,
  listAdminMcpServers,
  listMcpServers,
  listSkills,
  updateAdminMcpServer,
  updateAdminSkill,
  updateMcpServerEnabled,
  upsertAdminMcpServer,
  deleteAdminSkillOverride,
  updateSkillEnabled,
  updateUserToolEnabled,
} from "../../lib/backend-api";
import type {
  AdminMCPServerConfig,
  AdminMCPServerPayload,
  AdminSkillDetail,
  AdminSkillUpdatePayload,
  MCPServerAvailability,
  MCPServerTransport,
  Skill,
  ToolAvailability,
} from "../../lib/backend-types";
import { summarizeError } from "../../lib/format";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Switch } from "../ui/switch";
import { Textarea } from "../ui/textarea";

export interface ToolAccessSectionRef {
  refresh: () => void;
}

type ToolGroup = "builtin" | "integration";

const GROUP_META: Record<ToolGroup, { title: string; empty: string }> = {
  builtin: {
    title: "Встроенные инструменты",
    empty: "Для этого пользователя сейчас нет доступных встроенных инструментов.",
  },
  integration: {
    title: "Внешние интеграции",
    empty: "Сейчас нет доступных внешних интеграций.",
  },
};

export const ToolAccessSection = forwardRef<
  ToolAccessSectionRef,
  { onLoadingChange?: (loading: boolean) => void; isAdmin?: boolean }
>(function ToolAccessSection({ onLoadingChange, isAdmin }, ref) {
  const [tools, setTools] = useState<ToolAvailability[]>([]);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [mcpServers, setMcpServers] = useState<MCPServerAvailability[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savingKeys, setSavingKeys] = useState<Record<string, boolean>>({});
  const [savingSkillIds, setSavingSkillIds] = useState<Record<string, boolean>>({});
  const [savingMcpServerIds, setSavingMcpServerIds] = useState<Record<string, boolean>>({});
  const [editingSkill, setEditingSkill] = useState<AdminSkillDetail | null>(null);
  const [editingMcpServer, setEditingMcpServer] = useState<AdminMCPServerConfig | null>(null);
  const [isCreatingMcpServer, setIsCreatingMcpServer] = useState(false);
  const [isSavingEdit, setIsSavingEdit] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  useImperativeHandle(ref, () => ({ refresh: () => void loadTools(true) }));

  useEffect(() => {
    void loadTools();
  }, []);

  const grouped = useMemo(() => {
    const buckets: Record<ToolGroup, ToolAvailability[]> = {
      builtin: [],
      integration: [],
    };
    for (const tool of tools) {
      if (tool.kind === "integration") {
        buckets.integration.push(tool);
      } else {
        buckets.builtin.push(tool);
      }
    }
    return buckets;
  }, [tools]);

  async function loadTools(background = false): Promise<void> {
    if (background) {
      setIsRefreshing(true);
      onLoadingChange?.(true);
    } else {
      setIsLoading(true);
    }
    try {
      setError(null);
      const [toolsData, skillsData, mcpServersData] = await Promise.all([
        getUserTools(),
        listSkills(),
        listMcpServers(),
      ]);
      setTools(toolsData);
      setSkills(skillsData);
      setMcpServers(mcpServersData);
    } catch (loadError) {
      setError(summarizeError(loadError));
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
      onLoadingChange?.(false);
    }
  }

  async function handleToggleSkill(skill: Skill, nextEnabled: boolean): Promise<void> {
    setSavingSkillIds((prev) => ({ ...prev, [skill.skill_id]: true }));
    try {
      const updated = await updateSkillEnabled(skill.skill_id, nextEnabled);
      setSkills((prev) => prev.map((s) => (s.skill_id === updated.skill_id ? updated : s)));
      setError(null);
    } catch (toggleError) {
      setError(summarizeError(toggleError));
    } finally {
      setSavingSkillIds((prev) => {
        const next = { ...prev };
        delete next[skill.skill_id];
        return next;
      });
    }
  }

  async function handleToggleMcpServer(
    server: MCPServerAvailability,
    nextEnabled: boolean,
  ): Promise<void> {
    setSavingMcpServerIds((prev) => ({ ...prev, [server.server_id]: true }));
    try {
      const updated = await updateMcpServerEnabled(server.server_id, nextEnabled);
      setMcpServers((prev) => prev.map((item) => (item.server_id === updated.server_id ? updated : item)));
      setError(null);
    } catch (toggleError) {
      setError(summarizeError(toggleError));
    } finally {
      setSavingMcpServerIds((prev) => {
        const next = { ...prev };
        delete next[server.server_id];
        return next;
      });
    }
  }

  async function handleToggle(tool: ToolAvailability, nextEnabled: boolean): Promise<void> {
    setSavingKeys((prev) => ({ ...prev, [tool.tool_key]: true }));
    try {
      const updated = await updateUserToolEnabled(tool.tool_key, nextEnabled);
      setTools((prev) => prev.map((item) => (item.tool_key === updated.tool_key ? updated : item)));
      setError(null);
    } catch (toggleError) {
      setError(summarizeError(toggleError));
    } finally {
      setSavingKeys((prev) => {
        const next = { ...prev };
        delete next[tool.tool_key];
        return next;
      });
    }
  }

  async function handleEditSkill(skillId: string): Promise<void> {
    try {
      setEditError(null);
      const detail = await getAdminSkillDetail(skillId);
      setEditingSkill(detail);
    } catch (err) {
      setEditError(summarizeError(err));
    }
  }

  async function handleSaveEdit(payload: AdminSkillUpdatePayload): Promise<void> {
    if (!editingSkill) return;
    setIsSavingEdit(true);
    setEditError(null);
    try {
      await updateAdminSkill(editingSkill.skill_id, payload);
      setEditingSkill(null);
      await loadTools(true);
    } catch (err) {
      setEditError(summarizeError(err));
    } finally {
      setIsSavingEdit(false);
    }
  }

  async function handleResetOverride(): Promise<void> {
    if (!editingSkill) return;
    setIsSavingEdit(true);
    setEditError(null);
    try {
      await deleteAdminSkillOverride(editingSkill.skill_id);
      setEditingSkill(null);
      await loadTools(true);
    } catch (err) {
      setEditError(summarizeError(err));
    } finally {
      setIsSavingEdit(false);
    }
  }

  async function handleExportArchive(): Promise<void> {
    setIsExporting(true);
    try {
      const blob = await exportSkillsArchive();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "skills-export.zip";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(summarizeError(err));
    } finally {
      setIsExporting(false);
    }
  }

  async function handleCreateMcpServer(): Promise<void> {
    setEditError(null);
    setEditingMcpServer(null);
    setIsCreatingMcpServer(true);
  }

  async function handleEditMcpServer(serverId: string): Promise<void> {
    try {
      setEditError(null);
      const configs = await listAdminMcpServers();
      const config = configs.find((item) => item.server_id === serverId);
      if (!config) {
        throw new Error("MCP-сервер не найден");
      }
      setEditingMcpServer(config);
      setIsCreatingMcpServer(false);
    } catch (err) {
      setEditError(summarizeError(err));
    }
  }

  async function handleDeleteMcpServer(serverId: string): Promise<void> {
    setSavingMcpServerIds((prev) => ({ ...prev, [serverId]: true }));
    try {
      await deleteAdminMcpServer(serverId);
      await loadTools(true);
    } catch (err) {
      setError(summarizeError(err));
    } finally {
      setSavingMcpServerIds((prev) => {
        const next = { ...prev };
        delete next[serverId];
        return next;
      });
    }
  }

  async function handleSaveMcpServer(payload: AdminMCPServerPayload): Promise<void> {
    setIsSavingEdit(true);
    setEditError(null);
    try {
      if (editingMcpServer) {
        await updateAdminMcpServer(editingMcpServer.server_id, payload);
      } else {
        await upsertAdminMcpServer(payload);
      }
      setEditingMcpServer(null);
      setIsCreatingMcpServer(false);
      await loadTools(true);
    } catch (err) {
      setEditError(summarizeError(err));
    } finally {
      setIsSavingEdit(false);
    }
  }

  return (
    <div className="space-y-4">
      {error ? (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
          <AlertCircle className="mt-0.5 h-4 w-4 flex-none" />
          <span>{error}</span>
        </div>
      ) : null}

      {isLoading ? (
        <div className="rounded-xl border border-border/40 bg-background/30 px-4 py-6 text-sm text-muted-foreground">
          Загружаю список инструментов и интеграций...
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          <ToolGroupCard
            group="builtin"
            tools={grouped.builtin}
            savingKeys={savingKeys}
            onToggle={handleToggle}
          />
          <ToolGroupCard
            group="integration"
            tools={grouped.integration}
            savingKeys={savingKeys}
            onToggle={handleToggle}
          />
        </div>
      )}

      <McpServersCard
        servers={mcpServers}
        isLoading={isLoading}
        savingServerIds={savingMcpServerIds}
        onToggle={handleToggleMcpServer}
        isAdmin={isAdmin}
        onCreate={handleCreateMcpServer}
        onEdit={handleEditMcpServer}
        onDelete={handleDeleteMcpServer}
      />

      <SkillsCard
        skills={skills}
        isLoading={isLoading}
        savingSkillIds={savingSkillIds}
        onToggle={handleToggleSkill}
        isAdmin={isAdmin}
        onEdit={handleEditSkill}
        onExport={handleExportArchive}
        isExporting={isExporting}
      />

      <SkillEditModal
        skill={editingSkill}
        isSaving={isSavingEdit}
        error={editError}
        onSave={handleSaveEdit}
        onReset={handleResetOverride}
        onClose={() => { setEditingSkill(null); setEditError(null); }}
      />

      <McpServerEditModal
        server={editingMcpServer}
        isOpen={isCreatingMcpServer || Boolean(editingMcpServer)}
        isSaving={isSavingEdit}
        error={editError}
        onSave={handleSaveMcpServer}
        onClose={() => {
          setEditingMcpServer(null);
          setIsCreatingMcpServer(false);
          setEditError(null);
        }}
      />
    </div>
  );
});

function McpServersCard({
  servers,
  isLoading,
  savingServerIds,
  onToggle,
  isAdmin,
  onCreate,
  onEdit,
  onDelete,
}: {
  servers: MCPServerAvailability[];
  isLoading: boolean;
  savingServerIds: Record<string, boolean>;
  onToggle: (server: MCPServerAvailability, nextEnabled: boolean) => Promise<void>;
  isAdmin?: boolean;
  onCreate?: () => Promise<void>;
  onEdit?: (serverId: string) => Promise<void>;
  onDelete?: (serverId: string) => Promise<void>;
}) {
  const enabledCount = servers.filter((server) => server.effective_enabled).length;

  return (
    <div className="space-y-4 rounded-2xl border border-border/50 bg-background/30 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Server className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-bold uppercase tracking-[0.18em] text-muted-foreground">MCP-серверы</h4>
        </div>
        <div className="flex items-center gap-2">
          {!isLoading && servers.length > 0 && (
            <span className="text-xs text-muted-foreground">{enabledCount} / {servers.length} включено</span>
          )}
          {isAdmin && onCreate && (
            <Button variant="outline" size="sm" onClick={() => void onCreate()}>
              <Plus className="h-3.5 w-3.5" />
              <span className="ml-1 hidden sm:inline">Добавить</span>
            </Button>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="rounded-xl border border-dashed border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
          Загрузка MCP-серверов...
        </div>
      ) : servers.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
          MCP-серверы не настроены.
        </div>
      ) : (
        <div className="space-y-3">
          {servers.map((server) => (
            <McpServerRow
              key={server.server_id}
              server={server}
              isSaving={Boolean(savingServerIds[server.server_id])}
              onToggle={onToggle}
              isAdmin={isAdmin}
              onEdit={onEdit}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function McpServerRow({
  server,
  isSaving,
  onToggle,
  isAdmin,
  onEdit,
  onDelete,
}: {
  server: MCPServerAvailability;
  isSaving: boolean;
  onToggle: (server: MCPServerAvailability, nextEnabled: boolean) => Promise<void>;
  isAdmin?: boolean;
  onEdit?: (serverId: string) => Promise<void>;
  onDelete?: (serverId: string) => Promise<void>;
}) {
  const canToggle = server.enabled_globally && server.available_globally && !isSaving;
  const statusVariant = resolveMcpStatusVariant(server);
  const toolLabels = server.tools.slice(0, 6).map((tool) => tool.tool_name);

  return (
    <div className="rounded-xl border border-border/50 bg-secondary/20 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h5 className="text-sm font-semibold text-foreground">{server.name}</h5>
            <Badge variant={statusVariant}>{formatMcpStatusLabel(server)}</Badge>
            <Badge variant="outline" className="border-border/60 bg-background/50">
              {server.transport}
            </Badge>
          </div>
          {server.description ? (
            <p className="text-sm leading-relaxed text-muted-foreground">{server.description}</p>
          ) : null}
          {server.last_error ? (
            <p className="text-xs leading-relaxed text-destructive">{server.last_error}</p>
          ) : null}
          {toolLabels.length ? (
            <div className="flex flex-wrap gap-2">
              {toolLabels.map((toolName) => (
                <Badge key={toolName} variant="outline" className="border-border/60 bg-background/50">
                  {toolName}
                </Badge>
              ))}
              {server.tool_count > toolLabels.length ? (
                <Badge variant="outline" className="border-border/60 bg-background/50">
                  +{server.tool_count - toolLabels.length}
                </Badge>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            {isAdmin && onEdit && (
              <button
                onClick={() => void onEdit(server.server_id)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                title="Редактировать MCP-сервер"
              >
                <FilePenLine className="h-4 w-4" />
              </button>
            )}
            {isAdmin && onDelete && (
              <button
                onClick={() => void onDelete(server.server_id)}
                disabled={isSaving}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/15 hover:text-destructive transition-colors disabled:opacity-50"
                title="Удалить MCP-сервер"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            )}
            <Switch
              checked={server.enabled_for_user}
              disabled={!canToggle}
              aria-label={`Переключить MCP-сервер ${server.name}`}
              onCheckedChange={(checked) => {
                void onToggle(server, checked);
              }}
            />
          </div>
          {isSaving ? <span className="text-xs text-muted-foreground">Сохранение...</span> : null}
        </div>
      </div>
    </div>
  );
}

function McpServerEditModal({
  server,
  isOpen,
  isSaving,
  error,
  onSave,
  onClose,
}: {
  server: AdminMCPServerConfig | null;
  isOpen: boolean;
  isSaving: boolean;
  error: string | null;
  onSave: (payload: AdminMCPServerPayload) => Promise<void>;
  onClose: () => void;
}) {
  const [serverId, setServerId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [transport, setTransport] = useState<MCPServerTransport>("streamable_http");
  const [url, setUrl] = useState("");
  const [command, setCommand] = useState("");
  const [argsText, setArgsText] = useState("");
  const [envText, setEnvText] = useState("{}");
  const [timeoutSec, setTimeoutSec] = useState("30");
  const [enabled, setEnabled] = useState(true);
  const [enabledByDefault, setEnabledByDefault] = useState(true);

  useEffect(() => {
    if (!isOpen) return;
    setServerId(server?.server_id ?? "");
    setName(server?.name ?? "");
    setDescription(server?.description ?? "");
    setTransport(server?.transport ?? "streamable_http");
    setUrl(server?.url ?? "");
    setCommand(server?.command ?? "");
    setArgsText((server?.args ?? []).join("\n"));
    setEnvText(JSON.stringify(server?.env ?? {}, null, 2));
    setTimeoutSec(String(server?.timeout_sec ?? 30));
    setEnabled(server?.enabled ?? true);
    setEnabledByDefault(server?.enabled_by_default ?? true);
  }, [isOpen, server]);

  if (!isOpen) return null;

  const title = server ? `Редактирование MCP-сервера: ${server.name}` : "Добавление MCP-сервера";
  const canSave = Boolean(
    serverId.trim() &&
      name.trim() &&
      (transport === "stdio" ? command.trim() : url.trim()),
  );

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            Настройка MCP-сервера доступна только администраторам. Пользователи могут только включать или выключать настроенные серверы.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-none" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid gap-4 py-2">
          <div className="grid gap-2">
            <Label htmlFor="mcp-server-id">ID сервера</Label>
            <Input
              id="mcp-server-id"
              value={serverId}
              disabled={Boolean(server)}
              onChange={(event) => setServerId(event.target.value)}
              placeholder="finance-research"
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="mcp-name">Название</Label>
            <Input id="mcp-name" value={name} onChange={(event) => setName(event.target.value)} />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="mcp-description">Описание</Label>
            <Textarea
              id="mcp-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              rows={2}
            />
          </div>

          <div className="grid gap-2 sm:grid-cols-2">
            <div className="grid gap-2">
              <Label>Транспорт</Label>
              <Select value={transport} onValueChange={(value) => setTransport(value as MCPServerTransport)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="streamable_http">Потоковый HTTP</SelectItem>
                  <SelectItem value="stdio">стандартный ввод/вывод</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="mcp-timeout">Таймаут, секунды</Label>
              <Input
                id="mcp-timeout"
                type="number"
                min={1}
                max={300}
                value={timeoutSec}
                onChange={(event) => setTimeoutSec(event.target.value)}
              />
            </div>
          </div>

          {transport === "streamable_http" ? (
            <div className="grid gap-2">
              <Label htmlFor="mcp-url">URL</Label>
              <Input
                id="mcp-url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="http://127.0.0.1:8765/mcp"
              />
            </div>
          ) : (
            <>
              <div className="grid gap-2">
                <Label htmlFor="mcp-command">Команда</Label>
                <Input
                  id="mcp-command"
                  value={command}
                  onChange={(event) => setCommand(event.target.value)}
                  placeholder="python"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="mcp-args">Аргументы</Label>
                <Textarea
                  id="mcp-args"
                  value={argsText}
                  onChange={(event) => setArgsText(event.target.value)}
                  rows={4}
                  className="font-mono text-xs"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="mcp-env">Переменные окружения JSON</Label>
                <Textarea
                  id="mcp-env"
                  value={envText}
                  onChange={(event) => setEnvText(event.target.value)}
                  rows={5}
                  className="font-mono text-xs"
                />
              </div>
            </>
          )}

          <div className="flex flex-wrap gap-4 rounded-xl border border-border/50 bg-secondary/20 p-3">
            <label className="flex items-center gap-2 text-sm">
              <Switch checked={enabled} onCheckedChange={setEnabled} />
              Включено глобально
            </label>
            <label className="flex items-center gap-2 text-sm">
              <Switch checked={enabledByDefault} onCheckedChange={setEnabledByDefault} />
              Включено по умолчанию
            </label>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <DialogClose asChild>
            <Button variant="outline">Отмена</Button>
          </DialogClose>
          <Button
            disabled={!canSave || isSaving}
            onClick={() => {
              void onSave({
                server_id: serverId.trim(),
                name: name.trim(),
                description: description.trim() || null,
                transport,
                url: transport === "streamable_http" ? url.trim() : null,
                command: transport === "stdio" ? command.trim() : null,
                args: argsText.split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
                env: parseEnvJson(envText),
                timeout_sec: Number(timeoutSec) || 30,
                enabled,
                enabled_by_default: enabledByDefault,
              });
            }}
          >
            {isSaving ? "Сохранение..." : "Сохранить"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function SkillEditModal({
  skill,
  isSaving,
  error,
  onSave,
  onReset,
  onClose,
}: {
  skill: AdminSkillDetail | null;
  isSaving: boolean;
  error: string | null;
  onSave: (payload: AdminSkillUpdatePayload) => Promise<void>;
  onReset: () => Promise<void>;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [triggersText, setTriggersText] = useState("");
  const [coreMarkdown, setCoreMarkdown] = useState("");
  const [detailsMarkdown, setDetailsMarkdown] = useState("");

  useEffect(() => {
    if (skill) {
      setName(skill.name);
      setDescription(skill.description);
      setTriggersText(skill.triggers.join(", "));
      setCoreMarkdown(skill.core_markdown);
      setDetailsMarkdown(skill.details_markdown ?? "");
    }
  }, [skill]);

  if (!skill) return null;

  const hasChanges =
    name !== skill.name ||
    description !== skill.description ||
    triggersText !== skill.triggers.join(", ") ||
    coreMarkdown !== skill.core_markdown ||
    detailsMarkdown !== (skill.details_markdown ?? "");

  return (
    <Dialog open={!!skill} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Редактирование навыка: {skill.name}</DialogTitle>
          <DialogDescription>
            {skill.is_overridden
              ? "Навык имеет локальные изменения. Они будут применены немедленно."
              : "Изменения будут сохранены в базу данных и применены без перезапуска."}
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-none" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid gap-4 py-2">
          <div className="grid gap-2">
            <Label htmlFor="skill-name">Название</Label>
            <Input id="skill-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="skill-desc">Описание</Label>
            <Textarea id="skill-desc" value={description} onChange={(e) => setDescription(e.target.value)} rows={2} />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="skill-triggers">Триггеры (через запятую)</Label>
            <Input id="skill-triggers" value={triggersText} onChange={(e) => setTriggersText(e.target.value)} />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="skill-core">SKILL.md (основной файл)</Label>
            <Textarea
              id="skill-core"
              value={coreMarkdown}
              onChange={(e) => setCoreMarkdown(e.target.value)}
              rows={12}
              className="font-mono text-xs leading-relaxed"
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="skill-details">DETAILS.md (опционально)</Label>
            <Textarea
              id="skill-details"
              value={detailsMarkdown}
              onChange={(e) => setDetailsMarkdown(e.target.value)}
              rows={6}
              className="font-mono text-xs leading-relaxed"
            />
          </div>
        </div>

        <DialogFooter className="gap-2 sm:justify-between">
          <div>
            {skill.is_overridden && (
              <Button variant="destructive" disabled={isSaving} onClick={onReset}>
                Сбросить до заводских
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            <DialogClose asChild>
              <Button variant="outline">Отмена</Button>
            </DialogClose>
            <Button
              disabled={!hasChanges || isSaving}
              onClick={() =>
                onSave({
                  name: name !== skill.name ? name : undefined,
                  description: description !== skill.description ? description : undefined,
                  triggers: triggersText !== skill.triggers.join(", ")
                    ? triggersText.split(",").map((s) => s.trim()).filter(Boolean)
                    : undefined,
                  core_markdown: coreMarkdown !== skill.core_markdown ? coreMarkdown : undefined,
                  details_markdown:
                    detailsMarkdown !== (skill.details_markdown ?? "") ? detailsMarkdown || null : undefined,
                })
              }
            >
              {isSaving ? "Сохраняю..." : "Сохранить"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function SkillsCard({
  skills,
  isLoading,
  savingSkillIds,
  onToggle,
  isAdmin,
  onEdit,
  onExport,
  isExporting,
}: {
  skills: Skill[];
  isLoading: boolean;
  savingSkillIds: Record<string, boolean>;
  onToggle: (skill: Skill, nextEnabled: boolean) => Promise<void>;
  isAdmin?: boolean;
  onEdit?: (skillId: string) => Promise<void>;
  onExport?: () => Promise<void>;
  isExporting?: boolean;
}) {
  const enabledCount = skills.filter((s) => s.enabled_for_user).length;

  return (
    <div className="space-y-4 rounded-2xl border border-border/50 bg-background/30 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-bold uppercase tracking-[0.18em] text-muted-foreground">Навыки</h4>
        </div>
        <div className="flex items-center gap-2">
          {!isLoading && skills.length > 0 && (
            <span className="text-xs text-muted-foreground">{enabledCount} / {skills.length} включено</span>
          )}
          {isAdmin && (
            <Button variant="outline" size="sm" disabled={isExporting} onClick={onExport} title="Скачать архив">
              <Download className="h-3.5 w-3.5" />
              <span className="ml-1 hidden sm:inline">Архив</span>
            </Button>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="rounded-xl border border-dashed border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
          Загружаю навыки...
        </div>
      ) : skills.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
          Навыки не найдены. Добавьте файлы SKILL.md в папку <code className="font-mono text-xs">skills/</code>.
        </div>
      ) : (
        <div className="space-y-3">
          {skills.map((skill) => (
            <SkillRow
              key={skill.skill_id}
              skill={skill}
              isSaving={Boolean(savingSkillIds[skill.skill_id])}
              onToggle={onToggle}
              isAdmin={isAdmin}
              onEdit={onEdit}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function SkillRow({
  skill,
  isSaving,
  onToggle,
  isAdmin,
  onEdit,
}: {
  skill: Skill;
  isSaving: boolean;
  onToggle: (skill: Skill, nextEnabled: boolean) => Promise<void>;
  isAdmin?: boolean;
  onEdit?: (skillId: string) => Promise<void>;
}) {
  return (
    <div className="rounded-xl border border-border/50 bg-secondary/20 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h5 className="text-sm font-semibold text-foreground">{skill.name}</h5>
            <Badge variant="outline" className="border-primary/30 bg-primary/8 text-primary text-[10px]">
              аналитика
            </Badge>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">{skill.description}</p>
          {skill.triggers.length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-0.5">
              {skill.triggers.slice(0, 6).map((trigger) => (
                <span
                  key={trigger}
                  className="inline-flex items-center rounded-full border border-border/50 bg-background/40 px-2 py-0.5 text-[11px] text-muted-foreground"
                >
                  {trigger}
                </span>
              ))}
              {skill.triggers.length > 6 && (
                <span className="inline-flex items-center rounded-full border border-border/40 bg-background/30 px-2 py-0.5 text-[11px] text-muted-foreground">
                  +{skill.triggers.length - 6}
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            {isAdmin && onEdit && (
              <button
                onClick={() => onEdit(skill.skill_id)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                title="Редактировать навык"
              >
                <FilePenLine className="h-4 w-4" />
              </button>
            )}
            <Switch
              checked={skill.enabled_for_user}
              disabled={isSaving}
              aria-label={`Переключить навык ${skill.name}`}
              onCheckedChange={(checked) => {
                void onToggle(skill, checked);
              }}
            />
          </div>
          {isSaving ? <span className="text-xs text-muted-foreground">Сохраняю...</span> : null}
        </div>
      </div>
    </div>
  );
}

function ToolGroupCard({
  group,
  tools,
  savingKeys,
  onToggle,
}: {
  group: ToolGroup;
  tools: ToolAvailability[];
  savingKeys: Record<string, boolean>;
  onToggle: (tool: ToolAvailability, nextEnabled: boolean) => Promise<void>;
}) {
  const meta = GROUP_META[group];

  return (
    <div className="space-y-4 rounded-2xl border border-border/50 bg-background/30 p-4">
      <div>
        <h4 className="text-sm font-bold uppercase tracking-[0.18em] text-muted-foreground">
          {meta.title}
        </h4>
      </div>

      {tools.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
          {meta.empty}
        </div>
      ) : (
        <div className="space-y-3">
          {tools.map((tool) => (
            <ToolRow
              key={tool.tool_key}
              tool={tool}
              isSaving={Boolean(savingKeys[tool.tool_key])}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ToolRow({
  tool,
  isSaving,
  onToggle,
}: {
  tool: ToolAvailability;
  isSaving: boolean;
  onToggle: (tool: ToolAvailability, nextEnabled: boolean) => Promise<void>;
}) {
  const canToggle = tool.enabled_globally && tool.available_globally && !isSaving;
  const statusVariant = resolveStatusVariant(tool);
  const metaBits = [
    tool.source_type ? `источник: ${formatToolSourceTypeLabel(tool.source_type)}` : null,
    typeof tool.timeout_hint_sec === "number" ? `таймаут ~${tool.timeout_hint_sec} сек` : null,
  ].filter(Boolean);

  return (
    <div className="rounded-xl border border-border/50 bg-secondary/20 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h5 className="text-sm font-semibold text-foreground">{getToolLabel(tool)}</h5>
            <Badge variant={statusVariant}>{formatStatusLabel(tool)}</Badge>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">{getToolDescription(tool)}</p>
          {tool.capabilities.length ? (
            <div className="flex flex-wrap gap-2">
              {tool.capabilities.map((item) => (
                <Badge key={item} variant="outline" className="border-border/60 bg-background/50">
                  {item}
                </Badge>
              ))}
            </div>
          ) : null}
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {metaBits.map((item) => (
              <span
              key={item}
              className="inline-flex items-center gap-1 rounded-full border border-border/50 bg-background/40 px-2.5 py-1"
            >
                <PlugZap className="h-3.5 w-3.5" />
                {item}
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <Switch
            checked={tool.enabled_for_user}
            disabled={!canToggle}
            aria-label={`Переключить ${tool.tool_label}`}
            onCheckedChange={(checked) => {
              void onToggle(tool, checked);
            }}
          />
          {isSaving ? <span className="text-xs text-muted-foreground">Сохраняю...</span> : null}
        </div>
      </div>
    </div>
  );
}

function resolveMcpStatusVariant(
  server: MCPServerAvailability,
): "default" | "secondary" | "destructive" | "outline" {
  if (!server.available_globally) {
    return "destructive";
  }
  if (!server.enabled_globally) {
    return "outline";
  }
  if (server.effective_enabled) {
    return "default";
  }
  return "secondary";
}

function formatMcpStatusLabel(server: MCPServerAvailability): string {
  const normalized = server.status.trim().toLowerCase();
  if (!server.available_globally) {
    return normalized === "disabled" ? "отключено глобально" : "недоступно";
  }
  if (!server.enabled_globally) {
    return "отключено глобально";
  }
  if (!server.enabled_for_user) {
    return "отключено для пользователя";
  }
  if (normalized === "available") {
    return "доступно";
  }
  if (normalized === "enabled") {
    return "включено";
  }
  return "включено";
}

function parseEnvJson(raw: string): Record<string, string> {
  const clean = raw.trim();
  if (!clean) {
    return {};
  }
  try {
    const parsed = JSON.parse(clean) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).map(([key, value]) => [key, String(value)]),
    );
  } catch {
    return {};
  }
}

function resolveStatusVariant(
  tool: ToolAvailability,
): "default" | "secondary" | "destructive" | "outline" {
  if (!tool.available_globally) {
    return "destructive";
  }
  if (!tool.enabled_globally) {
    return "outline";
  }
  if (tool.status === "enabled" || tool.effective_enabled) {
    return "default";
  }
  return "secondary";
}

function formatStatusLabel(tool: ToolAvailability): string {
  const normalized = tool.status.trim().toLowerCase();
  if (!tool.available_globally) {
    return "недоступно";
  }
  if (!tool.enabled_globally) {
    return "отключено глобально";
  }
  if (!tool.enabled_for_user) {
    return "отключено для пользователя";
  }
  if (normalized === "enabled") {
    return "включено";
  }
  if (normalized === "available") {
    return "доступно";
  }
  if (normalized === "misconfigured") {
    return "ошибка конфигурации";
  }
  return "включено";
}

function getToolLabel(tool: ToolAvailability): string {
  return tool.display_name_ru?.trim() || tool.tool_label;
}

function getToolDescription(tool: ToolAvailability): string {
  return tool.description_ru?.trim() || tool.description;
}

function formatToolSourceTypeLabel(sourceType: string): string {
  const normalized = sourceType.trim().toLowerCase();
  if (normalized === "builtin") {
    return "встроенный";
  }
  if (normalized === "integration") {
    return "интеграция";
  }
  if (normalized === "mcp") {
    return "MCP";
  }
  return sourceType;
}

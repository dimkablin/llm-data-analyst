import { useEffect, useMemo, useState } from "react";
import { AlertCircle, PlugZap, Wrench } from "lucide-react";
import { getUserTools, updateUserToolEnabled } from "../../lib/backend-api";
import type { ToolAvailability } from "../../lib/backend-types";
import { summarizeError } from "../../lib/format";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Switch } from "../ui/switch";

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

export function ToolAccessSection() {
  const [tools, setTools] = useState<ToolAvailability[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savingKeys, setSavingKeys] = useState<Record<string, boolean>>({});

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
    } else {
      setIsLoading(true);
    }
    try {
      setError(null);
      setTools(await getUserTools());
    } catch (loadError) {
      setError(summarizeError(loadError));
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }

  async function handleToggle(tool: ToolAvailability, nextEnabled: boolean): Promise<void> {
    // For the merged "Веб поиск" row, toggle both underlying tools
    if (tool.tool_key === WEB_SEARCH_KEY) {
      const targets = grouped.integration.filter(
        (t) => WEB_SEARCH_TOOL_KEYS.has(t.tool_key) && t.enabled_globally && t.available_globally,
      );
      setSavingKeys((prev) => ({ ...prev, [WEB_SEARCH_KEY]: true }));
      try {
        await Promise.all(
          targets.map(async (t) => {
            const updated = await updateUserToolEnabled(t.tool_key, nextEnabled);
            setTools((prev) => prev.map((item) => (item.tool_key === updated.tool_key ? updated : item)));
          }),
        );
        setError(null);
      } catch (toggleError) {
        setError(summarizeError(toggleError));
      } finally {
        setSavingKeys((prev) => {
          const next = { ...prev };
          delete next[WEB_SEARCH_KEY];
          return next;
        });
      }
      return;
    }

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

  return (
    <div className="space-y-4 rounded-2xl border border-border/50 bg-secondary/25 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-primary/10 p-1.5 text-primary">
              <Wrench className="h-4 w-4" />
            </span>
            <h3 className="text-lg font-bold tracking-tight">Доступ к инструментам</h3>
          </div>
          <p className="max-w-3xl text-sm text-muted-foreground">
            Выберите, какие инструменты агенту разрешено использовать для этого аккаунта.
          </p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => void loadTools(true)}
          disabled={isLoading || isRefreshing}
        >
          Обновить
        </Button>
      </div>

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
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
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
    tool.requires_session_data ? "нужны данные сессии" : "работает без данных сессии",
    tool.source_type ? `источник: ${tool.source_type}` : null,
    typeof tool.timeout_hint_sec === "number" ? `timeout ~${tool.timeout_hint_sec} c` : null,
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
            aria-label={`Toggle ${tool.tool_label}`}
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
  return tool.status || "включено";
}

function getToolLabel(tool: ToolAvailability): string {
  return tool.display_name_ru?.trim() || tool.tool_label;
}

function getToolDescription(tool: ToolAvailability): string {
  return tool.description_ru?.trim() || tool.description;
}

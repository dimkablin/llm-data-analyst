import { ArrowRight, Check, Database, FolderKanban, Shield, Sparkles, Wrench } from "lucide-react";

function DiagramCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: JSX.Element;
}) {
  return (
    <div className="relative overflow-hidden rounded-[32px] border border-border/50 bg-card/90 p-5 shadow-[0_18px_50px_rgba(15,23,42,0.06)] backdrop-blur-md md:p-7">
      <div className="pointer-events-none absolute inset-x-10 top-0 h-24 rounded-full bg-primary/8 blur-3xl" />
      <div className="mb-5">
        <h3 className="text-xl font-bold tracking-tight">{title}</h3>
        {subtitle ? <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{subtitle}</p> : null}
      </div>
      <div className="overflow-x-auto rounded-[24px] border border-border/40 bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.12),_transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01))] p-4 dark:bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.18),_transparent_34%),linear-gradient(180deg,rgba(15,23,42,0.78),rgba(2,6,23,0.86))] md:p-6">
        {children}
      </div>
    </div>
  );
}

function Node({
  title,
  subtitle,
  accent = "default",
}: {
  title: string;
  subtitle?: string;
  accent?: "default" | "soft" | "strong";
}) {
  const tone =
    accent === "strong"
      ? "border-blue-400/80 bg-blue-500/15 shadow-[0_0_32px_rgba(59,130,246,0.14)]"
      : accent === "soft"
        ? "border-sky-400/60 bg-sky-500/10"
        : "border-blue-300/45 bg-background/60";

  return (
    <div className={`rounded-[22px] border px-5 py-4 text-center backdrop-blur-md ${tone}`}>
      <div className="text-[15px] font-semibold leading-tight text-foreground">{title}</div>
      {subtitle ? <div className="mt-2 text-[13px] leading-tight text-muted-foreground">{subtitle}</div> : null}
    </div>
  );
}

function Diamond({
  title,
  subtitle,
}: {
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="relative h-[132px] w-[132px] rotate-45 rounded-[26px] border border-blue-400/60 bg-sky-500/10 shadow-[0_0_30px_rgba(59,130,246,0.12)]">
      <div className="absolute inset-0 flex -rotate-45 flex-col items-center justify-center px-4 text-center">
        <div className="text-[14px] font-semibold leading-tight text-foreground">{title}</div>
        {subtitle ? <div className="mt-2 text-[12px] leading-tight text-muted-foreground">{subtitle}</div> : null}
      </div>
    </div>
  );
}

function LineArrow({ vertical = false }: { vertical?: boolean }) {
  return (
    <div className={`flex items-center justify-center ${vertical ? "h-10" : "w-12 min-w-12"}`}>
      <div
        className={`relative ${vertical ? "h-full w-px bg-blue-300/60" : "h-px w-full bg-blue-300/60"}`}
      >
        <div
          className={`absolute rounded-full bg-blue-300/90 ${vertical ? "-bottom-0.5 left-1/2 h-2 w-2 -translate-x-1/2" : "-right-0.5 top-1/2 h-2 w-2 -translate-y-1/2"}`}
        />
      </div>
    </div>
  );
}

function FlowPill({
  icon: Icon,
  label,
}: {
  icon: typeof Shield;
  label: string;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-blue-400/35 bg-blue-500/8 px-3 py-1.5 text-[12px] font-medium text-muted-foreground">
      <Icon className="h-3.5 w-3.5 text-blue-400" />
      <span>{label}</span>
    </div>
  );
}

export function ArchitectureDiagram() {
  return (
    <DiagramCard
      title="Схема компонентов"
      subtitle="Фронтенд управляет пользовательским сценарием, сервисный слой FastAPI держит продуктовые гарантии, обобщённая среда LangGraph выполняет агентный цикл, слой расширений добавляет доменные возможности."
    >
      <div className="min-w-[1060px] space-y-6">
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <FlowPill icon={Sparkles} label="REST / SSE-поток" />
          <FlowPill icon={Database} label="QueryExecutionService" />
          <FlowPill icon={Wrench} label="Типизированные инструменты + MCP" />
        </div>

        <div className="grid grid-cols-[170px_36px_190px_36px_220px_36px_220px_36px_210px_36px_180px] items-center gap-y-5">
          <Node title="Пользователь" subtitle="Браузер" />
          <LineArrow />
          <Node title="Фронтенд" subtitle="Пользовательский слой React" accent="soft" />
          <LineArrow />
          <Node title="Маршруты FastAPI" subtitle="Тонкая граница API" />
          <LineArrow />
          <Node title="QueryExecutionService" subtitle="Вход, источники, сохранение" accent="strong" />
          <LineArrow />
          <Node title="AgentRunner" subtitle="Среда LangGraph" accent="soft" />
          <LineArrow />

          <div className="space-y-4">
            <Node title="Типизированные инструменты" subtitle="Pandas, Plotly, SQL, RAG" />
            <Node title="Навыки / MCP" subtitle="Доменные расширения" />
            <Node title="LLM-провайдер" subtitle="OpenAI-совместимый" />
          </div>
        </div>
      </div>
    </DiagramCard>
  );
}

export function AgentCycleDiagram() {
  return (
    <DiagramCard
      title="LangGraph/ReAct цикл"
      subtitle="Граф остаётся маленьким и доменно-нейтральным: подготовка контекста, типизированный цикл инструментов и финальная оркестрация ответа."
    >
      <div className="min-w-[1020px] space-y-6">
        <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
          <FlowPill icon={Sparkles} label="prepare_context -> agent -> finalize" />
          <FlowPill icon={Wrench} label="События выполнения + типизированные артефакты" />
        </div>

        <div className="grid grid-cols-[170px_40px_220px_40px_190px_40px_220px_40px_160px] items-center">
          <Node title="Запрос" subtitle="Пользователя" />
          <LineArrow />
          <Node title="prepare_context" subtitle="источники, навыки, инструменты" accent="soft" />
          <LineArrow />
          <Node title="agent" subtitle="LLM-цикл инструментов" accent="soft" />
          <LineArrow />
          <Node title="типизированные инструменты" subtitle="сообщения, события, артефакты" />
          <LineArrow />
          <Node title="finalize" subtitle="Ответ" accent="strong" />
        </div>

        <div className="rounded-[24px] border border-blue-400/25 bg-background/50 p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
            <Check className="h-4 w-4 text-blue-400" />
            Граница среды выполнения
          </div>
          <div className="grid grid-cols-4 gap-4 text-sm text-muted-foreground">
            <div className="rounded-2xl border border-border/40 bg-background/60 p-4">Построитель контекста собирает метаданные выполнения без ключевых обходных путей.</div>
            <div className="rounded-2xl border border-border/40 bg-background/60 p-4">LLM вызывает только разрешённые типизированные инструменты.</div>
            <div className="rounded-2xl border border-border/40 bg-background/60 p-4">Инструменты возвращают нормализованные артефакты и события.</div>
            <div className="rounded-2xl border border-border/40 bg-background/60 p-4">Финализация формирует ответ без доменных шаблонов в исполнителе.</div>
          </div>
        </div>
      </div>
    </DiagramCard>
  );
}

export function SecurityDiagram() {
  return (
    <DiagramCard
      title="Вход и проверка владельца"
      subtitle="Каждый доступ к сессии проходит через валидацию токена и проверку владельца. Это изолирует пользовательские данные без сложного UI-слоя."
    >
      <div className="min-w-[1060px] space-y-6">
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <FlowPill icon={Shield} label="Вход по токену" />
          <FlowPill icon={FolderKanban} label="Владение сессией" />
        </div>

        <div className="grid grid-cols-[170px_40px_170px_56px_180px_56px_220px_56px_180px] items-center">
          <Node title="Вход" />
          <LineArrow />
          <Node title="Токен" subtitle="Доступа" />
          <LineArrow />
          <Diamond title="API" subtitle="запрос" />
          <LineArrow />
          <div className="space-y-5">
            <Node title="Пользователь" subtitle="определен" accent="soft" />
            <Node title="401" subtitle="Не авторизован" />
          </div>
          <LineArrow />
          <div className="space-y-5">
            <Diamond title="Владелец" subtitle="сессии?" />
            <div className="grid grid-cols-2 gap-4">
              <Node title="Чтение / запись" subtitle="сессии" accent="strong" />
              <Node title="404" subtitle="Не найдено" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4 text-sm">
          <div className="rounded-2xl border border-border/40 bg-background/60 p-4 text-muted-foreground">
            Валидный токен допускает запрос к контроллеру бэкенда.
          </div>
          <div className="rounded-2xl border border-border/40 bg-background/60 p-4 text-muted-foreground">
            Невалидный токен сразу завершает сценарий с `401`.
          </div>
          <div className="rounded-2xl border border-border/40 bg-background/60 p-4 text-muted-foreground">
            Для валидного пользователя выполняется проверка владельца сессии.
          </div>
          <div className="rounded-2xl border border-border/40 bg-background/60 p-4 text-muted-foreground">
            Только владелец получает доступ к чтению и записи данных.
          </div>
        </div>
      </div>
    </DiagramCard>
  );
}

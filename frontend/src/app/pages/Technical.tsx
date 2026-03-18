import { motion } from "motion/react";
import {
  Activity,
  CheckCircle2,
  Code2,
  Cpu,
  Database,
  Globe,
  Layers,
  Radar,
  Server,
  Shield,
  Terminal,
  Workflow,
  Zap,
  Braces,
  LockKeyhole,
  Boxes,
  Rocket,
  Bot,
  Wrench,
  Network,
} from "lucide-react";

import { Navigation } from "../components/Navigation";
import {
  AgentCycleDiagram,
  ArchitectureDiagram,
  SecurityDiagram,
} from "../components/TechnicalDiagrams";

const layerCards = [
  {
    id: "frontend",
    label: "Frontend",
    badge: "React + Vite",
    icon: Globe,
    color: "text-blue-400",
    bg: "bg-blue-400/10",
    items: [
      "Новый frontend является основной оболочкой продукта и объединяет рабочий сценарий, демо и архитектурные страницы.",
      "Workspace связывает чат, артефакты, dashboard и настройки без смены контекста.",
      "SSE-стриминг показывает ответ по мере генерации и поддерживает reasoning-события.",
      "UI сохраняет и рабочие, и future-ready сценарии без удаления демонстрационных блоков.",
    ],
  },
  {
    id: "backend",
    label: "Backend",
    badge: "FastAPI + Python",
    icon: Server,
    color: "text-emerald-400",
    bg: "bg-emerald-400/10",
    items: [
      "Backend отвечает за auth, ownership-check, session state, загрузку CSV и сериализацию результата.",
      "Ключевые endpoints покрывают sessions, data upload, query, query/stream, evaluate и auth/settings.",
      "TTL сессии по умолчанию 7 дней, лимит датасета по умолчанию 100 MB.",
      "Ответ возвращается в едином payload с текстом, артефактами и метриками.",
    ],
  },
  {
    id: "agent",
    label: "AI Agent",
    badge: "Reason-Action Loop",
    icon: Cpu,
    color: "text-violet-400",
    bg: "bg-violet-400/10",
    items: [
      "Агент работает в цикле Plan → Act → Observe → Decide с max-steps и timeout на действие.",
      "Простые запросы могут завершаться без инструментов, аналитические сценарии идут через tool-driven контур.",
      "Артефакты нормализуются в единый контракт plot, table или value.",
      "LLM-провайдер переключается между Ollama и vLLM без смены frontend-контракта.",
    ],
  },
];

const lifecycle = [
  {
    title: "Контекст сессии",
    sub: "Auth + dataset + history",
    text: "Пользователь открывает сессию, backend поднимает историю, данные и ранее собранные артефакты.",
  },
  {
    title: "Запрос в API",
    sub: "query / query/stream",
    text: "Frontend отправляет вопрос вместе с флагами use_history и include_reasoning.",
  },
  {
    title: "Планирование",
    sub: "Plan → route",
    text: "Агент определяет тип запроса и выбирает следующий шаг анализа.",
  },
  {
    title: "Инструментальный шаг",
    sub: "Act → Observe",
    text: "Backend выполняет действие, проверяет результат и превращает его в артефакты для UI.",
  },
  {
    title: "Финализация",
    sub: "SSE final payload",
    text: "Поток отдает токены и reasoning, затем фиксирует итоговый ответ, метрики и артефакты в сессии.",
  },
];

const guarantees = [
  "Ответ подтверждается артефактами, а не только текстовой генерацией.",
  "Ownership-check защищает доступ к данным и сессиям на каждом запросе.",
  "Лимиты, таймауты и max-steps делают агентный сценарий предсказуемым.",
  "Переход от локального Ollama-стенда к vLLM не ломает UI-контракт.",
];

const numberedSections = [
  {
    id: "components",
    number: "1",
    title: "Компоненты системы",
    icon: Boxes,
    iconColor: "text-blue-400",
    iconBg: "bg-blue-400/10",
    intro:
      "Система состоит из двух основных слоев: frontend ведет UX, стриминг и визуальный контур, backend берет на себя auth, session state, агентную оркестрацию и выдачу результата.",
    diagram: <ArchitectureDiagram />,
  },
  {
    id: "agent-cycle",
    number: "2",
    title: "Reason-Action агентный цикл",
    icon: Workflow,
    iconColor: "text-violet-400",
    iconBg: "bg-violet-400/10",
    intro:
      "Аналитический сценарий строится не как один непрозрачный ответ, а как итеративная цепочка шагов. Это повышает объяснимость и делает путь к выводу воспроизводимым.",
    diagram: <AgentCycleDiagram />,
    bullets: [
      "Streaming-ответ доставляет текст пользователю сразу, без ожидания полного завершения пайплайна.",
      "Инструментальный слой ограничен по времени выполнения и числу шагов.",
      "При проблеме на отдельном шаге система может вернуть fallback-ответ вместо немого сбоя.",
    ],
  },
  {
    id: "security",
    number: "3",
    title: "Безопасность и управление доступом",
    icon: LockKeyhole,
    iconColor: "text-rose-400",
    iconBg: "bg-rose-400/10",
    intro:
      "Каждая сессия привязана к владельцу. Любая операция чтения или записи проходит через bearer auth и ownership-check, что исключает случайный cross-user доступ.",
    diagram: <SecurityDiagram />,
    bullets: [
      "Пароли хешируются через scrypt с солью.",
      "Токены хранятся как хеши и работают с TTL.",
      "Доступ к данным, истории и артефактам проверяется на backend-стороне.",
    ],
  },
  {
    id: "state",
    number: "4",
    title: "Хранение состояния и артефактов",
    icon: Layers,
    iconColor: "text-amber-400",
    iconBg: "bg-amber-400/10",
    intro:
      "История чата, датасет, метрики и артефакты живут в backend session storage и восстанавливаются в интерфейсе при открытии или переключении чата.",
    bullets: [
      "Session TTL по умолчанию составляет 7 дней.",
      "Dataset limit по умолчанию составляет 100 MB.",
      "Финальный payload уже подготовлен к рендерингу, закреплению в dashboard и экспорту.",
    ],
  },
  {
    id: "deployment",
    number: "5",
    title: "Deployment и migration path",
    icon: Rocket,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-400/10",
    intro:
      "Локальный стенд уже годится для демо и пилота. При переходе в production меняется runtime на backend-слое, а frontend продолжает работать через тот же контракт.",
    bullets: [
      "Локальный режим использует Ollama.",
      "Production path предполагает перенос на vLLM без переписывания UI.",
      "Docker Compose позволяет воспроизводимо поднимать окружение целиком.",
    ],
  },
  {
    id: "llm",
    number: "6",
    title: "ML / LLM слой",
    icon: Bot,
    iconColor: "text-emerald-400",
    iconBg: "bg-emerald-400/10",
    intro:
      "Платформа работает через единый backend-контур, который маршрутизирует запросы к LLM-провайдеру и держит под контролем latency, reasoning-режим и ограничения по токенам.",
    cards: [
      {
        title: "Model Endpoint",
        text: "Backend ходит в OpenAI-compatible endpoint, поэтому может переключаться между Ollama и vLLM без изменения frontend.",
      },
      {
        title: "Thinking + Streaming",
        text: "При include_reasoning=true backend отдает reasoning-события и reasoning_token в SSE-потоке.",
      },
      {
        title: "Единый контур",
        text: "Один API-слой контролирует chat, analysis, историю, лимиты и итоговый формат ответа.",
      },
      {
        title: "Fallback safety",
        text: "Даже при частичном сбое шага пользователь получает результат в контролируемом виде, а не сломанный UX.",
      },
    ],
  },
  {
    id: "decision-making",
    number: "7",
    title: "Как агент принимает решения",
    icon: Cpu,
    iconColor: "text-violet-400",
    iconBg: "bg-violet-400/10",
    intro:
      "Перед запуском анализа система определяет тип запроса и решает, нужен ли инструментальный сценарий. Это снижает лишние вызовы и делает поведение более предсказуемым.",
    bullets: [
      "Route: классификация запроса на chat или analysis.",
      "Plan: выбор следующего шага и подходящего инструмента.",
      "Act: выполнение Python-tool с лимитом времени.",
      "Observe: интерпретация результата шага.",
      "Finalize: сбор финального ответа с артефактами и метриками.",
    ],
  },
  {
    id: "tooling",
    number: "8",
    title: "Tooling и контракт артефактов",
    icon: Wrench,
    iconColor: "text-amber-400",
    iconBg: "bg-amber-400/10",
    intro:
      "Инструменты вызываются как структурированные backend-функции и возвращают нормализованный результат. Это позволяет одинаково надежно рендерить графики, таблицы и значения в UI.",
    cards: [
      {
        title: "DataFrame Analytics",
        text: "Фильтрация, агрегации, описательная статистика и выборочные срезы по колонкам.",
      },
      {
        title: "Plot Builder",
        text: "Подготовка Plotly-совместимых данных для интерактивных графиков и dashboard-артефактов.",
      },
      {
        title: "Table Builder",
        text: "Возврат таблиц в детерминированном формате для рендера и экспорта.",
      },
      {
        title: "Value Extractor",
        text: "Числовые метрики и KPI для быстрых ответов и summary-блоков.",
      },
    ],
    code: `{
  "schema_version": "1.0",
  "artifact_type": "plot | table | value",
  "items": { "name": "<payload>" }
}`,
  },
  {
    id: "api",
    number: "9",
    title: "API для LLM-контура",
    icon: Braces,
    iconColor: "text-blue-400",
    iconBg: "bg-blue-400/10",
    intro:
      "Frontend не обращается к модели напрямую. Вся оркестрация идет через backend API, который контролирует auth, ownership, историю, лимиты и SSE-поток.",
    code: `Core endpoints:
POST   /sessions
GET    /sessions/{session_id}
POST   /sessions/{session_id}/data
POST   /sessions/{session_id}/query
POST   /sessions/{session_id}/query/stream
POST   /sessions/{session_id}/evaluate

Auth endpoints:
POST   /auth/register
POST   /auth/login
GET    /auth/me
GET    /auth/settings
PATCH  /auth/settings
POST   /auth/logout

Admin endpoints:
GET    /admin/users
POST   /admin/users
PATCH  /admin/users/{id}
DELETE /admin/users/{id}`,
  },
  {
    id: "limits",
    number: "10",
    title: "Ограничения, качество и roadmap",
    icon: Network,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-400/10",
    intro:
      "Платформа уже хорошо работает в интерактивной аналитике, но зрелый production-контур требует профилирования на ваших сценариях и формализации eval-практик.",
    bullets: [
      "Оптимальный режим сейчас, интерактивная аналитика в пределах 100 MB и ограниченного времени вычислений.",
      "Качество ответа зависит от структуры датасета, полноты колонок и точности вопроса.",
      "Для production нужен замер latency и throughput на реальных кейсах заказчика.",
      "Следующий уровень зрелости, eval-наборы, регресс-тесты промптов и scoring correctness/tool-use.",
    ],
  },
];

function SectionCard({
  number,
  title,
  icon: Icon,
  iconColor,
  iconBg,
  intro,
  diagram,
  bullets,
  cards,
  code,
}: (typeof numberedSections)[number]) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="rounded-[36px] border border-border/50 bg-card/90 p-8 shadow-[0_18px_50px_rgba(15,23,42,0.05)] backdrop-blur-md lg:p-10"
    >
      <div className="mb-6 flex items-start gap-4">
        <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-lg font-bold text-primary">
          {number}
        </div>
        <div>
          <div className="mb-3 flex items-center gap-3">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">{title}</h2>
            <div className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-2xl ${iconBg}`}>
              <Icon className={`h-5 w-5 ${iconColor}`} />
            </div>
          </div>
          <p className="mt-3 max-w-4xl text-[16px] leading-relaxed text-muted-foreground">{intro}</p>
        </div>
      </div>

      {diagram ? <div className="mb-6">{diagram}</div> : null}

      {bullets ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {bullets.map((item) => (
            <div key={item} className="flex items-start gap-3 rounded-2xl border border-border/40 bg-background/60 p-5">
              <CheckCircle2 className="mt-0.5 h-5 w-5 flex-shrink-0 text-primary" />
              <p className="text-[15px] leading-relaxed text-muted-foreground">{item}</p>
            </div>
          ))}
        </div>
      ) : null}

      {cards ? (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {cards.map((card) => (
            <article key={card.title} className="rounded-2xl border border-border/40 bg-background/60 p-6">
              <h3 className="mb-3 text-xl font-bold tracking-tight">{card.title}</h3>
              <p className="text-[15px] leading-relaxed text-muted-foreground">{card.text}</p>
            </article>
          ))}
        </div>
      ) : null}

      {code ? (
        <pre className="mt-6 overflow-x-auto rounded-[24px] border border-border/40 bg-background/80 p-6 text-sm leading-relaxed text-muted-foreground">
          <code>{code}</code>
        </pre>
      ) : null}
    </motion.section>
  );
}

export function Technical() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />

      <main className="max-w-[1320px] mx-auto px-8 py-20">
        <div className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-primary"
          >
            <Terminal className="h-3.5 w-3.5" />
            <span className="text-[11px] font-bold uppercase tracking-widest">System Architecture</span>
          </motion.div>

          <h1 className="mb-6 text-5xl font-bold tracking-tight md:text-6xl">Архитектура</h1>
          <p className="max-w-3xl text-xl leading-relaxed text-muted-foreground">
            Инженерное объяснение платформы: как устроены UI и backend, как работает агентный
            контур, как обеспечивается безопасность и почему система готова к production-сценариям.
          </p>
        </div>

        <div className="mb-16 rounded-[36px] border border-border/40 bg-secondary/25 p-8 lg:p-10">
          <h2 className="mb-8 text-3xl font-bold tracking-tight">Ключевые гарантии платформы</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {guarantees.map((item) => (
              <div key={item} className="flex items-start gap-3 rounded-2xl border border-border/40 bg-card/80 p-5">
                <CheckCircle2 className="mt-0.5 h-5 w-5 flex-shrink-0 text-primary" />
                <p className="text-[15px] leading-relaxed text-muted-foreground">{item}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mb-16 grid grid-cols-1 gap-8 lg:grid-cols-3">
          {layerCards.map((layer, i) => (
            <motion.div
              key={layer.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              className="group rounded-[32px] border border-border/50 bg-card p-8 transition-all hover:border-primary/30"
            >
              <div className="mb-8 flex items-start justify-between">
                <div className={`rounded-2xl p-4 ${layer.bg} ${layer.color} transition-transform group-hover:scale-110`}>
                  <layer.icon className="h-6 w-6" />
                </div>
                <div className="rounded-full border border-border/50 bg-secondary px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                  {layer.badge}
                </div>
              </div>
              <h3 className="mb-6 text-2xl font-bold">{layer.label}</h3>
              <ul className="space-y-4">
                {layer.items.map((item) => (
                  <li key={item} className="flex items-start gap-3 text-[14px] leading-relaxed text-muted-foreground">
                    <CheckCircle2 className={`mt-1 h-4 w-4 flex-shrink-0 ${layer.color}`} />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        <div className="mb-16 rounded-[40px] border border-border/40 bg-secondary/30 p-8 lg:p-12">
          <h2 className="mb-12 flex items-center gap-3 text-3xl font-bold">
            <Code2 className="h-8 w-8 text-primary" />
            Request Lifecycle
          </h2>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-5">
            {lifecycle.map((step, index) => (
              <div key={step.title} className="rounded-2xl border border-border/50 bg-card p-6">
                <div className="mb-4 flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 text-sm font-bold text-primary">
                  {index + 1}
                </div>
                <div className="mb-1 text-[14px] font-bold">{step.title}</div>
                <div className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                  {step.sub}
                </div>
                <p className="text-[13px] leading-relaxed text-muted-foreground">{step.text}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-10">
          {numberedSections.map((section) => (
            <SectionCard key={section.id} {...section} />
          ))}
        </div>
      </main>
    </div>
  );
}

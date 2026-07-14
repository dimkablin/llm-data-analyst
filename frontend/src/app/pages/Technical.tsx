import { motion } from "motion/react";
import {
  CheckCircle2,
  Code2,
  Cpu,
  Globe,
  Layers,
  Server,
  Terminal,
  Workflow,
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
    label: "Фронтенд",
    badge: "React + Vite",
    icon: Globe,
    color: "text-blue-400",
    bg: "bg-blue-400/10",
    items: [
      "Фронтенд является пользовательским слоем: экраны, рабочие сценарии, настройки, представление артефактов и типизированное потребление API.",
      "Рабочая область связывает чат, источники, артефакты, доску и настройки без доступа к внутренним деталям среды выполнения.",
      "SSE-стриминг показывает токены, рассуждение, события инструментов, фазы и граф выполнения по мере работы.",
      "Фронтенд синхронизируется с бэкендом через стабильные DTO и значения перечислений, а не через внутренние состояния исполнителя.",
    ],
  },
  {
    id: "backend",
    label: "Бэкенд",
    badge: "FastAPI + Python",
    icon: Server,
    color: "text-emerald-400",
    bg: "bg-emerald-400/10",
    items: [
      "Маршруты FastAPI остаются тонкими и делегируют сценарии запроса и потоковой выдачи в QueryExecutionService.",
      "Сервисный слой проверяет аутентификацию, владельца, пользовательские настройки, выбранные навыки, инструменты, источники и сохранение состояния.",
      "Pydantic-схемы фиксируют контракты фронтенда и бэкенда для запросов, ответов, потоковой отдачи и артефактов.",
      "Бэкенд отвечает за продуктовую инфраструктуру: сессии, хранилище, наблюдаемость, аутентификацию, управление источниками и права доступа.",
    ],
  },
  {
    id: "agent",
    label: "Среда LangGraph",
    badge: "Generic ReAct",
    icon: Cpu,
    color: "text-violet-400",
    bg: "bg-violet-400/10",
    items: [
      "AgentRunner является точкой сборки и запускает граф prepare_context → agent → finalize.",
      "prepare_context собирает обобщённый контекст: метаданные сессии, источника, инструмента и навыка без ключевых обходных путей.",
      "agent выполняет цикл LLM-вызовов инструментов через типизированные инструменты, права доступа и события выполнения.",
      "finalize собирает финальный ответ и проверяет контракты, не подменяя хорошие ответы доменными шаблонами.",
    ],
  },
];

const lifecycle = [
  {
    title: "Контекст сессии",
    sub: "Вход + источники + история",
    text: "Пользователь открывает сессию, бэкенд поднимает владельца, источники, историю и ранее собранные артефакты.",
  },
  {
    title: "Запрос в API",
    sub: "query / query/stream",
    text: "Фронтенд отправляет типизированный запрос с вопросом, флагами истории и рассуждения, а также пользовательскими настройками.",
  },
  {
    title: "Сервисная граница",
    sub: "QueryExecutionService",
    text: "Сервис проверяет владельца, права инструментов, выбранные навыки, контекст источников и настройку среды выполнения.",
  },
  {
    title: "Среда LangGraph",
    sub: "prepare → agent → finalize",
    text: "Граф готовит контекст, выполняет цикл инструментов и финализирует ответ с артефактами и эффектами выполнения.",
  },
  {
    title: "Финализация",
    sub: "Финальная нагрузка SSE",
    text: "Потоковая отдача отправляет события, бэкенд сохраняет сообщения и артефакты, UI показывает финальную типизированную нагрузку.",
  },
];

const guarantees = [
  "Обобщённая среда LangGraph не содержит клиентской или доменной жёсткой привязки и остаётся переиспользуемым движком.",
  "Точки FastAPI тонкие: бизнес-логика живёт в сервисах и сценариях использования, а не в обработчиках маршрутов.",
  "Типизированные инструменты, Pydantic-схемы, права доступа и контракты артефактов фиксируют границу среды выполнения.",
  "Доменное поведение подключается через навыки, доменные манифесты и MCP-адаптеры, а не через ветки исполнителя.",
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
      "Система разделена на пользовательский слой фронтенда, продуктовый слой бэкенда, обобщённую среду выполнения LangGraph и слой расширений. Каждый слой имеет свой контракт и не должен подменять ответственность соседнего слоя.",
    diagram: <ArchitectureDiagram />,
  },
  {
    id: "agent-cycle",
    number: "2",
    title: "LangGraph/ReAct цикл",
    icon: Workflow,
    iconColor: "text-violet-400",
    iconBg: "bg-violet-400/10",
    intro:
      "Текущий граф остаётся компактным: prepare_context собирает обобщённый контекст выполнения, agent выполняет цикл вызовов инструментов, finalize отвечает за финальную оркестрацию ответа.",
    diagram: <AgentCycleDiagram />,
    bullets: [
      "Состояние выполнения, маршрутизация, события, права доступа и ссылки на артефакты остаются явными структурами.",
      "Инструментальный слой ограничен правами доступа, таймаутом и типизированными схемами входа/выхода.",
      "Доменные восстановления или детерминированные сводки не должны появляться в runner.py.",
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
      "Каждая сессия привязана к владельцу. Любая операция чтения или записи проходит через токен доступа и проверку владельца, что исключает случайный межпользовательский доступ.",
    diagram: <SecurityDiagram />,
    bullets: [
      "Пароли хешируются через scrypt с солью.",
      "Токены хранятся как хеши и работают с TTL.",
      "Доступ к данным, истории и артефактам проверяется на стороне бэкенда.",
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
      "История чата, источники данных, эффекты выполнения и артефакты живут в хранилище сессий бэкенда и восстанавливаются через типизированные контракты фронтенда.",
    bullets: [
      "Session TTL по умолчанию составляет 7 дней.",
      "Состояние источника отделено от представления UI и внутренних деталей среды выполнения.",
      "Финальная полезная нагрузка подготовлена к рендерингу, закреплению на доске, экспорту и последующему восстановлению.",
    ],
  },
  {
    id: "deployment",
    number: "5",
    title: "Развёртывание и путь миграции",
    icon: Rocket,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-400/10",
    intro:
      "Контур Docker/nginx держит фронтенд как единую точку входа. При переходе между LLM-провайдерами или целями развёртывания фронтенд продолжает работать через тот же контракт бэкенда.",
    bullets: [
      "Локальный режим может использовать Ollama или другой OpenAI-совместимый адрес.",
      "Промышленный контур может использовать vLLM/TGI/OpenAI-совместимого провайдера без переписывания UI.",
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
      "Платформа работает через единый контур бэкенда, который маршрутизирует запросы к LLM-провайдеру и держит под контролем задержку, режим рассуждения, лимиты токенов и потоковые обратные вызовы.",
    cards: [
      {
        title: "Адрес модели",
        text: "Бэкенд обращается к OpenAI-совместимому адресу, поэтому может переключаться между Ollama и vLLM без изменения фронтенда.",
      },
      {
        title: "Рассуждение и поток",
        text: "При include_reasoning=true бэкенд отдаёт события и токены рассуждения в SSE-потоке.",
      },
      {
        title: "Единый контур",
        text: "Один сервисный слой контролирует настройку запроса, историю, источники, политику инструментов, сохранение состояния и итоговый формат ответа.",
      },
      {
        title: "Безопасное восстановление",
        text: "Резервные ответы формируются на границе API и сервисов и не должны превращаться в доменные шаблоны внутри обобщённой среды выполнения.",
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
      "Решения агента должны проходить через обобщённое состояние выполнения, права инструментов и цикл LLM/инструментов. Клиентские ветки и ключевые обходные пути не являются частью текущего контракта среды выполнения.",
    bullets: [
      "prepare_context: сбор источников, разрешённых инструментов, выбранных навыков и метаданных выполнения.",
      "agent: цикл LLM-вызовов инструментов с типизированными инструментами и правами доступа.",
      "Результат инструмента: нормализованные сообщения, артефакты и события выполнения.",
      "finalize: финальный ответ, предупреждения и проверка контрактов.",
      "Сервисный слой: сохранение состояния, API-ответ и жизненный цикл потоковой отдачи.",
    ],
  },
  {
    id: "tooling",
    number: "8",
    title: "Инструменты и контракт артефактов",
    icon: Wrench,
    iconColor: "text-amber-400",
    iconBg: "bg-amber-400/10",
    intro:
      "Инструменты вызываются через слой реестра и политики, имеют строгие схемы и возвращают нормализованный результат. Это позволяет одинаково надёжно отображать графики, таблицы, значения, заметки и доменные артефакты в UI.",
    cards: [
      {
        title: "Аналитика DataFrame",
        text: "Фильтрация, агрегации, описательная статистика и выборочные срезы по колонкам через контуры pandas/SQL.",
      },
      {
        title: "Построитель графиков",
        text: "Подготовка Plotly-совместимых данных для интерактивных графиков и артефактов доски.",
      },
      {
        title: "Построитель таблиц",
        text: "Возврат таблиц в детерминированном формате для рендера, экспорта и восстановления сессии.",
      },
      {
        title: "Адаптер расширений",
        text: "Слой MCP и доменных расширений связывает навык в разметке, права инструментов, схемы запросов и ответов, а также ожидаемые артефакты.",
      },
    ],
    code: `{
  "schema_version": "1.0",
  "type": "plot | table | value | note",
  "content": { "name": "<полезная-нагрузка>" },
  "metadata": { "source": "<инструмент-или-расширение>" }
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
      "Фронтенд не обращается к модели напрямую. Вся оркестрация идёт через API бэкенда и сервисный слой, которые контролируют аутентификацию, владельца, настройки, историю, контекст источников, политику инструментов и SSE-поток.",
    code: `Основные точки API:
POST   /sessions
GET    /sessions/{session_id}
POST   /sessions/{session_id}/data
POST   /sessions/{session_id}/query
POST   /sessions/{session_id}/query/stream
POST   /sessions/{session_id}/evaluate

Точки входа:
POST   /auth/register
POST   /auth/login
GET    /auth/me
GET    /auth/settings
PATCH  /auth/settings
POST   /auth/logout

Точки администратора:
GET    /admin/users
POST   /admin/users
PATCH  /admin/users/{id}
DELETE /admin/users/{id}`,
  },
  {
    id: "limits",
    number: "10",
    title: "Ограничения, качество и план развития",
    icon: Network,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-400/10",
    intro:
      "Платформа уже подходит для интерактивной аналитики, но зрелый промышленный контур требует профилирования на реальных источниках, eval-наборов и строгих регрессионных проверок промптов, инструментов и контрактов.",
    bullets: [
      "Оптимальный режим сейчас: интерактивная аналитика с контролируемыми лимитами датасета и выполнения инструментов.",
      "Качество ответа зависит от структуры датасета, полноты колонок и точности вопроса.",
      "Для промышленного запуска нужен замер задержки, пропускной способности, профиля памяти и надёжности инструментов на реальных кейсах.",
      "Следующий уровень зрелости: eval-наборы, регрессионные тесты промптов и инструментов, оценка корректности и использования инструментов.",
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
            <span className="text-[11px] font-bold uppercase tracking-widest">Архитектура системы</span>
          </motion.div>

          <h1 className="mb-6 text-5xl font-bold tracking-tight md:text-6xl">Архитектура</h1>
          <p className="max-w-3xl text-xl leading-relaxed text-muted-foreground">
            Инженерное объяснение текущей системы: тонкие маршруты FastAPI, QueryExecutionService,
            обобщённая среда LangGraph, типизированные инструменты, события выполнения, артефакты
            и граница доменных расширений.
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

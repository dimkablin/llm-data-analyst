import { RevealOnScroll } from "../components/RevealOnScroll";
import { SiteNav } from "../components/SiteNav";
import { SpotlightCard } from "../components/reactbits/SpotlightCard";
import { StarBorder } from "../components/reactbits/StarBorder";
import type { AuthUser } from "../types";

type LandingPageProps = {
  currentUser: AuthUser | null;
  onNavigate: (path: "/" | "/user" | "/technical" | "/app" | "/phoenix") => void;
};

const CORE_BENEFITS = [
  {
    title: "Ускорение принятия решений",
    text: "Руководители получают первый обоснованный ответ в течение одной сессии, а не после длинного цикла запросов в BI-команду."
  },
  {
    title: "Прозрачность и доверие",
    text: "Каждый вывод подтверждается артефактами: таблицами, графиками и кодом, который можно проверить и переиспользовать."
  },
  {
    title: "Снижение операционной нагрузки",
    text: "Часть типовых аналитических запросов уходит в self-service формат, не перегружая команду данных."
  },
  {
    title: "Production-ready контур",
    text: "Аутентификация, изоляция сессий, лимиты и fallback-механика позволяют запускать пилот без компромиссов по стабильности."
  }
];

const EXECUTIVE_OUTCOMES = [
  "Согласованный формат ответов для бизнеса, продукта и аналитики.",
  "Единая точка входа: чат, аналитика, артефакты и dashboard в одном интерфейсе.",
  "Быстрая подготовка к совещаниям: ключевые цифры и визуализации доступны сразу.",
  "Экспорт результатов в HTML для обсуждения с командой и руководством."
];

const BUSINESS_CASES = [
  {
    title: "Выручка и маржинальность",
    text: "Найдите драйверы роста и просадки по сегментам, продуктам и периодам без ручного построения десятков запросов."
  },
  {
    title: "Продуктовая аналитика",
    text: "Проверяйте гипотезы по удержанию, воронке и поведению пользователей в формате «вопрос -> ответ -> артефакт»."
  },
  {
    title: "Операционная эффективность",
    text: "Выявляйте узкие места в процессах и быстро обосновывайте управленческие действия на данных."
  }
];

const IMPLEMENTATION_FLOW = [
  {
    title: "Шаг 1. Загрузка данных",
    text: "Подключаете CSV и сразу начинаете работу в диалоге, без сложной настройки фронта или SQL в интерфейсе."
  },
  {
    title: "Шаг 2. Агентный анализ",
    text: "Система строит план анализа, вызывает инструменты и возвращает структурированные артефакты."
  },
  {
    title: "Шаг 3. Сборка dashboard",
    text: "Фиксируете ключевые графики и таблицы в визуальный слой для команды и стейкхолдеров."
  },
  {
    title: "Шаг 4. Принятие решения",
    text: "Экспортируете результат и используете его как основу для управленческого обсуждения."
  }
];

const PROOF_METRICS = [
  { value: "5-15 мин", label: "до первого содержательного ответа" },
  { value: "100 MB", label: "интерактивный лимит датасета" },
  { value: "SSE + Agent", label: "живой стриминг и контролируемый tool-loop" }
];

export function LandingPage({ currentUser, onNavigate }: LandingPageProps): JSX.Element {
  return (
    <div className="site-page landing-page landing-page-refresh">
      <SiteNav currentUser={currentUser} onNavigate={onNavigate} />

      <main className="landing-wrap landing-refresh-wrap">
        <RevealOnScroll className="landing-refresh-hero">
          <div className="landing-refresh-copy">
            <span className="landing-chip landing-chip-refresh">AI-платформа для управленческих решений</span>
            <h1>Превращайте данные в решения</h1>
            <p>
              «Генеративная аналитика» объединяет аналитический чат, агентный цикл и dashboard в одном
              интерфейсе. Руководители и команды получают не просто ответ, а проверяемую логику с цифрами,
              графиками и понятным обоснованием.
            </p>
            <div className="landing-proof-row landing-proof-row-refresh">
              <span>FastAPI + SSE streaming</span>
              <span>Reason-Action Agent Loop</span>
              <span>Plotly artifacts + export</span>
            </div>
            <div className="landing-actions landing-refresh-actions">
              <StarBorder
                as="button"
                type="button"
                className="landing-star-cta"
                onClick={() => onNavigate(currentUser ? "/app" : "/user")}
                color="rgba(56, 189, 248, 0.95)"
                speed="5s"
              >
                {currentUser ? "Открыть рабочую область" : "Запустить демо"}
              </StarBorder>
              <button type="button" className="btn-ghost" onClick={() => onNavigate("/technical")}>
                Архитектура и безопасность
              </button>
            </div>
          </div>

          <aside className="landing-refresh-summary">
            <h2>Что получает бизнес</h2>
            <ul>
              {EXECUTIVE_OUTCOMES.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
            <div className="landing-refresh-summary-note">
              Подходит для пилота и production-контура: от first-pass анализа до регулярных управленческих сессий.
            </div>
          </aside>
        </RevealOnScroll>

        <section className="landing-metrics landing-refresh-metrics">
          {PROOF_METRICS.map((metric, index) => (
            <RevealOnScroll key={metric.value} className="landing-metric-card landing-metric-card-refresh" delayMs={index * 70}>
              <h3>{metric.value}</h3>
              <p>{metric.label}</p>
            </RevealOnScroll>
          ))}
        </section>

        <section className="landing-refresh-story-grid">
          <RevealOnScroll className="landing-refresh-story-card" delayMs={40}>
            <h2>Проблема, которую мы закрываем</h2>
            <p>
              В большинстве компаний между вопросом руководителя и финальным ответом проходит слишком много времени:
              разные команды, разрозненные инструменты, потеря контекста и неполная прозрачность логики расчета.
            </p>
            <p>
              Это замедляет решение и снижает доверие к аналитике: цифры есть, но путь до них часто неочевиден.
            </p>
          </RevealOnScroll>

          <RevealOnScroll className="landing-refresh-story-card" delayMs={110}>
            <h2>Как решает платформа</h2>
            <p>
              Платформа строит единый процесс: вопрос в чате, агентный анализ, артефакты и собранный dashboard.
              В итоге бизнес получает быстрый и объяснимый результат, с которым можно идти на встречу уже сегодня.
            </p>
            <p>
              Модель работает в контролируемом контуре, а результаты остаются проверяемыми и воспроизводимыми.
            </p>
          </RevealOnScroll>
        </section>

        <section className="landing-feature-grid landing-feature-grid-refresh">
          {CORE_BENEFITS.map((feature, index) => (
            <RevealOnScroll key={feature.title} delayMs={90 + index * 70}>
              <SpotlightCard className="landing-feature-card landing-feature-card-refresh" spotlightColor="rgba(56, 189, 248, 0.26)">
                <h2>{feature.title}</h2>
                <p>{feature.text}</p>
              </SpotlightCard>
            </RevealOnScroll>
          ))}
        </section>

        <section className="landing-refresh-cases">
          <RevealOnScroll className="landing-refresh-section-head" delayMs={100}>
            <h2>Ключевые бизнес-сценарии</h2>
            <p>
              Ниже примеры направлений, где платформа дает быстрый эффект и помогает перейти от обсуждений к действиям.
            </p>
          </RevealOnScroll>

          <div className="landing-refresh-case-grid">
            {BUSINESS_CASES.map((item, index) => (
              <RevealOnScroll className="landing-refresh-case-card" delayMs={140 + index * 70} key={item.title}>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </RevealOnScroll>
            ))}
          </div>
        </section>

        <section className="landing-refresh-flow">
          <RevealOnScroll className="landing-refresh-section-head" delayMs={120}>
            <h2>Логика работы: от вопроса к действию</h2>
            <p>Прозрачный путь для бизнеса и аналитики, который легко объяснить на демо и в ежедневной работе.</p>
          </RevealOnScroll>
          <div className="landing-refresh-flow-grid">
            {IMPLEMENTATION_FLOW.map((item, index) => (
              <RevealOnScroll className="landing-refresh-flow-card" delayMs={160 + index * 60} key={item.title}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </RevealOnScroll>
            ))}
          </div>
        </section>

        <RevealOnScroll className="landing-refresh-final-cta" delayMs={220}>
          <h2>Готово к демонстрации и запуску пилота</h2>
          <p>
            Если вам нужно показать красивую, логичную и убедительную аналитику на данных уже в ближайшие дни,
            платформа закрывает этот сценарий end-to-end.
          </p>
          <div className="landing-actions">
            <button type="button" onClick={() => onNavigate(currentUser ? "/app" : "/user")}>
              {currentUser ? "Перейти в workspace" : "Начать сейчас"}
            </button>
            <button type="button" className="btn-ghost" onClick={() => onNavigate("/technical")}>
              Посмотреть технический контур
            </button>
          </div>
        </RevealOnScroll>
      </main>
    </div>
  );
}

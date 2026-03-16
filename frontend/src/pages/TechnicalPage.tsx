import {
  AgentCycleStaticDiagram,
  ArchitectureStaticDiagram,
  SecurityStaticDiagram
} from "../components/TechnicalDiagrams";
import { RevealOnScroll } from "../components/RevealOnScroll";
import { SpotlightCard } from "../components/reactbits/SpotlightCard";
import { StarBorder } from "../components/reactbits/StarBorder";
import { SiteNav } from "../components/SiteNav";
import type { AuthUser } from "../types";

type TechnicalPageProps = {
  currentUser: AuthUser | null;
  onNavigate: (path: "/" | "/user" | "/technical" | "/app" | "/phoenix") => void;
};

const TECH_STATS = [
  { value: "FastAPI + SSE", label: "backend-контракт для потокового ответа и финального payload" },
  { value: "Reason-Action", label: "итеративный агентный контур Plan -> Act -> Observe -> Finalize" },
  { value: "Session ownership", label: "изолированный доступ к истории, датасету и артефактам" }
];

const PLATFORM_GUARANTEES = [
  "Ответы с подтверждающими артефактами, а не только текстовой генерацией.",
  "Контролируемый tool-loop с ограничениями по времени и fallback-механикой.",
  "Изоляция пользовательских данных и проверка ownership на каждом запросе.",
  "Готовый путь от локального стенда к production inference без смены UI-контракта."
];

export function TechnicalPage({ currentUser, onNavigate }: TechnicalPageProps): JSX.Element {
  return (
    <div className="site-page technical-page technical-page-refresh">
      <SiteNav currentUser={currentUser} onNavigate={onNavigate} />

      <main className="technical-wrap-pro technical-wrap-refresh">
        <RevealOnScroll className="technical-hero technical-hero-refresh">
          <span className="landing-chip">Архитектура платформы</span>
          <h1>Технический контур платформы: понятно для бизнеса, прозрачно для инженеров</h1>
          <p>
            Ниже собрана инженерная карта системы: как устроены UI и backend, как агент принимает
            решения, как обеспечивается безопасность и почему этот контур подходит для production.
          </p>
          <div className="technical-hero-actions">
            <StarBorder
              as="button"
              type="button"
              className="technical-star-btn"
              onClick={() => onNavigate(currentUser ? "/app" : "/user")}
              color="rgba(56, 189, 248, 0.9)"
              speed="5s"
            >
              {currentUser ? "Открыть рабочую область" : "Запустить демо"}
            </StarBorder>
          </div>
        </RevealOnScroll>

        <section className="technical-kpis technical-kpis-refresh">
          {TECH_STATS.map((item, idx) => (
            <RevealOnScroll key={item.value} delayMs={40 + idx * 70}>
              <SpotlightCard className="technical-kpi-card technical-kpi-card-refresh" spotlightColor="rgba(56, 189, 248, 0.2)">
                <strong>{item.value}</strong>
                <span>{item.label}</span>
              </SpotlightCard>
            </RevealOnScroll>
          ))}
        </section>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={55}>
          <h2>Ключевые гарантии платформы</h2>
          <ul>
            {PLATFORM_GUARANTEES.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={80}>
          <h2>1) Компоненты системы</h2>
          <p>
            Система состоит из двух сервисов. Frontend отвечает за интерактивный UX, стриминг
            токенов, управление артефактами и состоянием пользователя. Backend обеспечивает
            аутентификацию, ownership-проверки сессий, загрузку датасета, выполнение агентного
            контура и сериализацию результатов.
          </p>
          <ArchitectureStaticDiagram />
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={110}>
          <h2>2) Reason-Action агентный цикл</h2>
          <p>
            Агент работает по итеративной схеме: гипотеза -&gt; выбор инструмента -&gt; выполнение
            -&gt; интерпретация -&gt; следующий шаг. Это обеспечивает воспроизводимость аналитики и
            прозрачность происхождения артефактов (таблица/график/метрика + код).
          </p>
          <AgentCycleStaticDiagram />
          <ul>
            <li>Поддерживается streaming-ответ с постепенной доставкой токенов.</li>
            <li>Инструментальный слой изолирован и имеет ограничения по времени выполнения.</li>
            <li>Падение конкретного шага не ломает UX: backend возвращает fallback-ответ.</li>
          </ul>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={140}>
          <h2>3) Безопасность и управление доступом</h2>
          <p>
            Каждая сессия привязана к владельцу. Любой доступ к данным/истории проходит bearer
            проверку и ownership check. Это исключает случайный cross-user доступ.
          </p>
          <SecurityStaticDiagram />
          <ul>
            <li>Пароли хешируются через `scrypt` с солью.</li>
            <li>Токены хранятся как хеши и имеют TTL.</li>
            <li>Операции с сессиями сохраняют метаданные активности.</li>
          </ul>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={170}>
          <h2>4) Хранение состояния и артефактов</h2>
          <p>
            История чата и артефактов живет в backend session storage и синхронизируется с UI при
            открытии/переключении чата. Во frontend дополнительно хранится пользовательский UI-state
            (тема, вид дашборда, порядок pinned артефактов).
          </p>
          <ul>
            <li>Session TTL: 7 дней (конфигурируемо).</li>
            <li>Dataset limit: 100 MB (задается через env).</li>
            <li>Экспорт дашборда и истории чата поддерживается в HTML.</li>
          </ul>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={200}>
          <h2>5) Deployment и migration path</h2>
          <p>
            Локальный режим использует Ollama. Production path предполагает перенос на vLLM без
            переписывания frontend-контракта: backend продолжает работать через OpenAI-compatible
            endpoint и существующий агентный контур.
          </p>
          <ul>
            <li>Docker Compose для воспроизводимого окружения.</li>
            <li>Изоляция frontend/backend сервисов для независимого масштабирования.</li>
            <li>Готовность к подключению observability и eval-контура.</li>
          </ul>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={230}>
          <h2>6) ML/LLM слой: модель, режимы и why-it-works</h2>
          <p>
            Платформа строится вокруг единой LLM, которая решает две задачи: разговорный ответ и
            инструментальный Reason-Action цикл для аналитики. Это упрощает эксплуатацию: один
            inference endpoint, один профиль latency/cost и единый контроль качества.
          </p>
          <div className="technical-key-grid">
            <article>
              <h3>Model Endpoint</h3>
              <p>
                Backend работает с OpenAI-compatible API (`/v1/chat/completions`), что позволяет
                без изменений UI переключаться между Ollama и vLLM.
              </p>
            </article>
            <article>
              <h3>Thinking + Streaming</h3>
              <p>
                Включены потоковые ответы (SSE) и reasoning-режим. Токены идут пользователю сразу,
                а финальный результат фиксируется в сессии вместе с артефактами.
              </p>
            </article>
            <article>
              <h3>Single-Model Strategy</h3>
              <p>
                Одна модель для chat/analysis уменьшает операционную сложность и риск рассинхрона
                поведения между «чатовой» и «инструментальной» моделями.
              </p>
            </article>
            <article>
              <h3>Fallback Safety</h3>
              <p>
                При сбое шага агент возвращает осмысленный fallback-ответ, поэтому пользователь
                получает ответ даже при ошибках инструментов или таймаутах.
              </p>
            </article>
          </div>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={260}>
          <h2>7) Агент: как принимаются решения и почему стабильно</h2>
          <p>
            Перед запуском анализа агент роутит запрос в режим `chat` или `analysis`. Простые
            диалоговые запросы обрабатываются без инструментов, а запросы по данным идут через
            tool-chain. Это снижает лишние вызовы и повышает предсказуемость.
          </p>
          <ul>
            <li>`Route`: классификация интента пользователя (`chat` / `analysis`).</li>
            <li>`Plan`: формирование следующего шага и выбора инструмента.</li>
            <li>`Act`: выполнение Python-tool с лимитом времени.</li>
            <li>`Observe`: интерпретация результата шага и решение о продолжении цикла.</li>
            <li>`Finalize`: синтез итогового ответа + прикрепление артефактов + метрики.</li>
          </ul>
          <p>
            Механизм стабильности: ограничение по итерациям, ограничение по времени шага,
            явная сериализация артефактов, контроль истории и строгий session ownership.
          </p>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={290}>
          <h2>8) Tooling: какие инструменты есть и их API-контракт</h2>
          <p>
            Инструменты изолированы в backend и вызываются агентом как структурированные функции.
            На входе - нормализованный запрос и текущее состояние; на выходе - артефакты единого
            формата (`table`, `plot`, `value`) и служебные метаданные.
          </p>
          <div className="technical-key-grid">
            <article>
              <h3>DataFrame Analytics</h3>
              <p>Фильтрация, агрегации, описательная статистика, feature-level срезы.</p>
            </article>
            <article>
              <h3>Plot Builder</h3>
              <p>Генерация Plotly JSON для интерактивных графиков с единым рендером в UI.</p>
            </article>
            <article>
              <h3>Table Builder</h3>
              <p>Возврат таблиц в `split`-формате для детерминированной сериализации/экспорта.</p>
            </article>
            <article>
              <h3>Value Extractor</h3>
              <p>Метрики/числа для быстрых KPI-ответов и компактных summary-блоков.</p>
            </article>
          </div>
          <pre className="technical-code">{`Artifact payload (normalized):
{
  "id": "artifact_...",
  "type": "plot | table | value",
  "text": "human-readable title",
  "data": {
    "format": "plotly-json | split | value",
    "data": { ... }
  },
  "meta": {
    "code": "...python that produced artifact"
  }
}`}</pre>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={320}>
          <h2>9) API для работы с LLM-контурами</h2>
          <p>
            Frontend не ходит напрямую в LLM. Вся оркестрация идет через backend API, который
            контролирует auth, ownership, историю, лимиты и формат ответа.
          </p>
          <pre className="technical-code">{`Core endpoints:
POST   /sessions
GET    /sessions
GET    /sessions/{id}
DELETE /sessions/{id}
PATCH  /sessions/{id}/title
POST   /sessions/{id}/title/generate
POST   /sessions/{id}/data
POST   /sessions/{id}/query
POST   /sessions/{id}/query/stream
POST   /sessions/{id}/evaluate

Auth endpoints:
POST   /auth/register
POST   /auth/login
GET    /auth/me
POST   /auth/change-password
GET    /auth/settings
PATCH  /auth/settings
POST   /auth/logout

Admin endpoints:
GET    /admin/users
POST   /admin/users
PATCH  /admin/users/{id}
DELETE /admin/users/{id}`}</pre>
          <p>
            Для стриминга используется SSE: сначала `start`, затем токены/события хода, в финале -
            итоговый payload с текстом, артефактами и метриками.
          </p>
        </RevealOnScroll>

        <RevealOnScroll className="technical-article-card technical-article-card-refresh" delayMs={350}>
          <h2>10) Ограничения, качество и roadmap</h2>
          <ul>
            <li>Текущий оптимум - интерактивная аналитика в пределах 100 MB и ограниченного времени вычислений.</li>
            <li>Качество ответа зависит от структуры датасета, полноты колонок и четкости вопроса.</li>
            <li>Для production рекомендуется vLLM + профилирование latency/token throughput на ваших сценариях.</li>
            <li>Следующий уровень зрелости: eval-наборы, регресс-тесты промптов, scoring correctness/tool-use.</li>
          </ul>
        </RevealOnScroll>
      </main>
    </div>
  );
}

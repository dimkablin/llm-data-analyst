import { RevealOnScroll } from "../components/RevealOnScroll";
import { SpotlightCard } from "../components/reactbits/SpotlightCard";
import { SiteNav } from "../components/SiteNav";
import type { AuthUser } from "../types";

type PhoenixPageProps = {
  currentUser: AuthUser;
  onNavigate: (path: "/" | "/user" | "/technical" | "/app" | "/phoenix") => void;
};

const OBSERVABILITY_VALUES = [
  {
    title: "Единая картина трассировки",
    text: "Видно, как агент проходил шаги, где вызывались инструменты и в какой точке формировался финальный ответ."
  },
  {
    title: "Контроль качества ответов",
    text: "Можно быстро находить проблемные участки reasoning/tool-calls и улучшать стабильность бизнес-сценариев."
  },
  {
    title: "Управление эксплуатацией",
    text: "Phoenix помогает принимать решения по оптимизации промптов, лимитов и конфигурации inference-контура."
  }
];

export function PhoenixPage({ currentUser, onNavigate }: PhoenixPageProps): JSX.Element {
  const iframeSrc = "/phoenix/";

  return (
    <div className="site-page phoenix-page phoenix-page-refresh">
      <SiteNav currentUser={currentUser} onNavigate={onNavigate} />

      <main className="technical-wrap-pro phoenix-wrap-refresh">
        <RevealOnScroll className="technical-hero phoenix-hero-refresh">
          <span className="landing-chip">Phoenix</span>
          <h1>Наблюдаемость агентного контура</h1>
          <p>
            Администраторская зона для контроля trace, tool-вызовов и метрик выполнения.
            Используйте ее как инженерный cockpit для повышения качества аналитических ответов.
          </p>
        </RevealOnScroll>

        <section className="phoenix-values-grid">
          {OBSERVABILITY_VALUES.map((item, index) => (
            <RevealOnScroll key={item.title} delayMs={80 + index * 70}>
              <SpotlightCard className="phoenix-value-card" spotlightColor="rgba(56, 189, 248, 0.22)">
                <h2>{item.title}</h2>
                <p>{item.text}</p>
              </SpotlightCard>
            </RevealOnScroll>
          ))}
        </section>

        <RevealOnScroll className="phoenix-frame-card" delayMs={180}>
          <iframe
            className="phoenix-frame"
            src={iframeSrc}
            title={`Phoenix (${currentUser.username})`}
            sandbox="allow-scripts allow-same-origin allow-forms"
          />
        </RevealOnScroll>
      </main>
    </div>
  );
}

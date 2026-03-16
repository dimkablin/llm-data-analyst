import type { AuthUser } from "../types";

type SiteNavProps = {
  currentUser: AuthUser | null;
  onNavigate: (path: "/" | "/user" | "/technical" | "/app" | "/phoenix") => void;
  className?: string;
};

export function SiteNav({
  currentUser,
  onNavigate,
  className
}: SiteNavProps): JSX.Element {
  const handleAuthClick = (): void => {
    onNavigate("/user");
  };

  return (
    <header className={`site-nav ${className || ""}`.trim()}>
      <button
        type="button"
        className="site-brand"
        onClick={() => onNavigate("/")}
        aria-label="Перейти на главную"
      >
        <span className="site-brand-mark" aria-hidden="true">GA</span>
        <span className="site-brand-meta">
          <span className="site-logo">Генеративная аналитика</span>
          <span className="site-tagline">Студия AI-решений</span>
        </span>
      </button>

      <div className="site-nav-links" role="navigation" aria-label="Главное меню">
        <button type="button" className="btn-ghost site-nav-btn" onClick={() => onNavigate("/")}>
          Платформа
        </button>
        <button
          type="button"
          className="btn-ghost site-nav-btn"
          onClick={() => onNavigate("/technical")}
        >
          Технический контур
        </button>
        <button
          type="button"
          className="btn-ghost site-nav-btn"
          onClick={() => onNavigate(currentUser ? "/app" : "/user")}
        >
          Рабочая область
        </button>
        <button type="button" className="btn-ghost site-nav-btn" onClick={handleAuthClick}>
          {currentUser ? "Аккаунт" : "Вход"}
        </button>
        {currentUser?.is_admin ? (
          <button
            type="button"
            className="btn-ghost site-nav-btn"
            onClick={() => onNavigate("/phoenix")}
          >
            Phoenix
          </button>
        ) : null}
        {currentUser ? (
          <span className="site-user-chip" title={`User: ${currentUser.username}`}>
            {currentUser.username}
          </span>
        ) : null}
      </div>
    </header>
  );
}

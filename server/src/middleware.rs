use actix_session::{ SessionMiddleware, Session };
use actix_session::config::{ BrowserSession, CookieContentSecurity };
use actix_session::storage::{ CookieSessionStore };
use actix_web::cookie::{ Cookie, Key, SameSite };

pub fn session_middleware() -> SessionMiddleware<CookieSessionStore> {
    SessionMiddleware::builder(
        CookieSessionStore::default(), Key::from(&[0; 64])
    )
    .cookie_secure(true)
    .session_lifecycle(BrowserSession::default())
    .cookie_same_site(SameSite::Strict)
    .cookie_content_security(CookieContentSecurity::Private)
    .cookie_http_only(true)
    .build()
}
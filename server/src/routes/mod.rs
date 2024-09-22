use actix_web::web;

pub mod user_routes;

// pub fn init(cfg: &mut web::ServiceConfig) {
//     cfg.service(web::resource("/users").route(web::get().to(get_users)));
// }

pub fn init(cfg: &mut web::ServiceConfig) {
    user_routes::user_routes(cfg);
}
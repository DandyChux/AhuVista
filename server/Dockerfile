ARG RUST_VERSION=1.75.0
ARG APP_NAME=ahuvista-server
ARG SERVER_PORT
ARG POSTGRES_HOST
ARG POSTGRES_USER
ARG POSTGRES_PASSWORD
ARG POSTGRES_PORT
ARG POSTGRES_DB
ARG DATABASE_URL
ARG JWT_EXPIRED_IN
ARG JWT_SECRET
ARG JWT_MAXAGE

################################################################################
# xx is a helper for cross-compilation.
# See https://github.com/tonistiigi/xx/ for more information.
FROM --platform=$BUILDPLATFORM tonistiigi/xx:1.3.0 AS xx

################################################################################
# Create a stage for building the application.
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-alpine AS build
ARG APP_NAME
WORKDIR /app

# Copy cross compilation utilities from the xx stage.
COPY --from=xx / /

# Install host build dependencies.
RUN apk add --no-cache clang lld musl-dev git file openssl-dev build-base

# This is the architecture you’re building for, which is passed in by the builder.
# Placing it here allows the previous steps to be cached across architectures.
ARG TARGETPLATFORM

# Install cross compilation build dependencies.
RUN xx-apk add --no-cache musl-dev gcc openssl-dev

# Build the application.
# Leverage a cache mount to /usr/local/cargo/registry/
# for downloaded dependencies, a cache mount to /usr/local/cargo/git/db
# for git repository dependencies, and a cache mount to /app/target/ for 
# compiled dependencies which will speed up subsequent builds.
# Leverage a bind mount to the src directory to avoid having to copy the
# source code into the container. Once built, copy the executable to an
# output directory before the cache mounted /app/target is unmounted.
COPY ./ ./server
# COPY ./server/Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock
# COPY ./common ./common
RUN echo "Source code has been copied."

ARG DATABASE_URL
ARG SERVER_PORT
ARG JWT_EXPIRED_IN
ARG JWT_SECRET
ARG JWT_MAXAGE
ARG POSTGRES_DB
ARG POSTGRES_USER
ARG POSTGRES_PASSWORD
ARG POSTGRES_HOST
ARG POSTGRES_PORT

# Set the environment variables for the server
ENV DATABASE_URL=$DATABASE_URL
ENV SERVER_PORT=$SERVER_PORT
ENV JWT_EXPIRED_IN=$JWT_EXPIRED_IN
ENV JWT_SECRET=$JWT_SECRET
ENV JWT_MAXAGE=$JWT_MAXAGE
ENV POSTGRES_DB=$POSTGRES_DB
ENV POSTGRES_USER=$POSTGRES_USER
ENV POSTGRES_PASSWORD=$POSTGRES_PASSWORD
ENV POSTGRES_HOST=$POSTGRES_HOST
ENV POSTGRES_PORT=$POSTGRES_PORT

# CD into the server directory
WORKDIR /app/server

RUN --mount=type=cache,target=/app/target/,id=rust-cache-${APP_NAME}-${TARGETPLATFORM} \
    --mount=type=cache,target=/usr/local/cargo/git/db \
    --mount=type=cache,target=/usr/local/cargo/registry/ \
    xx-cargo build --release --target-dir ./target && \
    cp ./target/$(xx-cargo --print-target-triple)/release/$APP_NAME /bin/server && \
    xx-verify /bin/server

################################################################################
# Create a new stage for running the application that contains the minimal
# runtime dependencies for the application. This often uses a different base
# image from the build stage where the necessary files are copied from the build
# stage.
#
# The example below uses the alpine image as the foundation for running the app.
# By specifying the "3.18" tag, it will use version 3.18 of alpine. If
# reproducability is important, consider using a digest
# (e.g., alpine@sha256:664888ac9cfd28068e062c991ebcff4b4c7307dc8dd4df9e728bedde5c449d91).
FROM alpine:3.18 AS final

# CD back to the app directory
WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser

# Copy the executable from the "build" stage.
COPY --from=build /bin/server /bin/

# Expose the port that the application listens on.
EXPOSE 8000

# What the container should run when it is started.
CMD ["/bin/server"]

################################################################################
# Create a stage for development.
# This stage uses cargo-watch to automatically reload the server during development.
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-alpine AS development
ARG APP_NAME
WORKDIR /app

# Copy cross compilation utilities from the xx stage.
COPY --from=xx / /

# Install host build dependencies.
RUN apk add --no-cache clang lld musl-dev git file openssl-dev build-base

# This is the architecture you’re building for, which is passed in by the builder.
# Placing it here allows the previous steps to be cached across architectures.
ARG TARGETPLATFORM

# Install cross compilation build dependencies.
RUN xx-apk add --no-cache musl-dev gcc

# Set RUSTFLAGS to use clang as the linker
ENV RUSTFLAGS="-C linker=clang"

# Install cargo-watch for hot-reloading during development.
RUN cargo install cargo-watch

# Copy your source code into the container.
COPY ./src ./src
# COPY ./common ./common
COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock
COPY ./migrations ./migrations
RUN echo "Source code has been copied."

# Expose the port that the application listens on.
EXPOSE 8000

# Use cargo-watch to run the application and reload it when changes are made.
CMD ["cargo", "watch", "-w", "./src", "-x", "run"]
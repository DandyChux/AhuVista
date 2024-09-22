# Path: Makefile
start-server:
	@echo "Starting server..."
	@cd server && docker-compose -f docker-compose.dev.yml up

# Path: Makefile
stop-server:
	@echo "Stopping server..."
	@cd server && docker-compose -f docker-compose.dev.yml down

# Path: Makefile
build-server:
	@echo "Building server..."
	@cd server && cargo build --release

# Path: Makefile
run-server:
	@echo "Running server..."
	@cd server && cargo run

# Path: Makefile
test-server:
	@echo "Testing server..."
	@cd server && cargo test

# Path: Makefile
build-web-client:
	@echo "Building web client..."
	@cd client && yarn expo build:web

# Path: Makefile
build-android-client:
	@echo "Building android client..."
	@cd client && yarn expo build:android

# Path: Makefile
build-ios-client:
	@echo "Building ios client..."
	@cd client && yarn expo build:ios

# Path: Makefile
run-client:
	@echo "Running client..."
	@cd client && yarn expo start

# Path: Makefile
start: start-server start-client

# Path: Makefile
build: build-server build-client

# Path: Makefile
run: run-server run-client

# Path: Makefile
test: test-server test-client
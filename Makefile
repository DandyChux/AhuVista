# Path: Makefile
start-server:
	@echo "Starting server..."
	@cd server && go run main.go

# Path: Makefile
build-server:
	@echo "Building server..."
	@go build -o server/bin/server server/main.go

# Path: Makefile
run-server:
	@echo "Running server..."
	@./server/bin/server

# Path: Makefile
test-server:
	@echo "Testing server..."
	@go test -v ./server/...

# Path: Makefile
clean-server:
	@echo "Cleaning server..."
	@rm -rf server/bin

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
test-client:
	@echo "Testing client..."
	@go test -v ./client/...

# Path: Makefile
clean-client:
	@echo "Cleaning client..."
	@rm -rf client/bin

# Path: Makefile
start: start-server start-client

# Path: Makefile
build: build-server build-client

# Path: Makefile
run: run-server run-client

# Path: Makefile
test: test-server test-client

# Path: Makefile
clean: clean-server clean-client

# Path: Makefile
rust-build:
	@echo "Building Rust client..."
	@cd rustml && cargo build --release
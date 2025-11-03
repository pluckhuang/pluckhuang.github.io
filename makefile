.PHONY: build
build:
	@bundle exec jekyll serve
install:
	@bundle install
install-ruby-brew:
	@echo "Installing Ruby via Homebrew..."
	@brew install ruby
install-bundler:
	@gem install bundler
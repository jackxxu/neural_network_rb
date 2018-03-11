
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "neural_network_rb/version"

Gem::Specification.new do |spec|
  spec.name          = "neural_network_rb"
  spec.version       = NeuralNetworkRb::VERSION
  spec.authors       = ["Jack Xu"]
  spec.email         = ["jackxxu@gmail.com"]

  spec.summary       = %q{neural network ruby implementaiton}
  spec.homepage      = "https://github.com/jackxxu/neural_network_rb"

  # Prevent pushing this gem to RubyGems.org. To allow pushes either set the 'allowed_push_host'
  # to allow pushing to a single host or delete this section to allow pushing to any host.
  if spec.respond_to?(:metadata)
    spec.metadata['allowed_push_host'] = "https://rubygems.org"
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end

  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "numo-narray", "~> 0.9.1.1"
  spec.add_dependency "numo-linalg", "~> 0.1.1"
  spec.add_dependency "chunky_png", "~> 1.3.10"

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rspec", "~> 3.0"
end

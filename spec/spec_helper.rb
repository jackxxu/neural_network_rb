require "bundler/setup"
require "neural_network_rb"
require 'pry'

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # config.filter_run :focus => true
end


module NArrayComparator 
  class << self
    def equal(exp, obs, delta = 1e-6)
      (exp.shape == obs.shape) && ((exp - obs).abs < delta).all?
    end
  end
end
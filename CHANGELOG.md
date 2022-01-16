# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Return `InferenceResult` to inference client from server
- Return result ndarrays from `InferenceClient` `forward`

## [0.0.2] - 2022-01-16
### Added
- `inference_server`
- Usage example to README

### Changed
- Set `InferenceServer` and `InferenceClient` as only public classes for `torchdemon` package

## [0.0.1] - 2022-01-14
### Added
- `inference_scheduler`
- `inference_client`
- `inference_model`
- `inference_queue`

[Unreleased]: https://github.com/jacknurminen/torchdemon/compare/0.0.2...master
[0.0.2]: https://github.com/jacknurminen/torchdemon/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/jacknurminen/torchdemon/tree/0.0.1

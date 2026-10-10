"""
Image downloader adapters.

Responsibility: fetch the bytes at an `models.ImageCandidate.url` and
persist them to a local scratch directory, returning a
`models.DownloadedImage`. Implements `interfaces.ImageDownloader`.

Conventions for future adapters added here:
    - A single adapter (e.g. an async HTTP-client-based `HttpImageDownloader`)
      typically suffices for every search provider, since download concerns
      (timeouts, retries, content-type/size validation) are independent of
      where the URL came from. Multiple adapters may still be useful for
      testing (a fake/in-memory downloader) or for sources that require
      special handling (authenticated CDNs, headless-browser rendering).
    - Adapters are responsible for: streaming the response, enforcing size/
      content-type limits, computing a checksum, and writing to
      `destination_dir` with a collision-safe filename.
    - Adapters must not perform any classification or business-logic
      decisions — that is the classifier's job.

No concrete downloader is implemented yet; this is scaffolding only.
"""

# images/

**Intentionally empty.** The Optera starter set is real client operational data —
phone photos of invoices, mechanic logs and dashboards carrying vendor names,
GSTINs, phone numbers, signatures and vehicle registrations. The dataset README
that shipped with it asks that it not be redistributed, and a public repository
would be exactly that.

To run the pipeline, drop the 47 provided files here:

```
images/
  optera_doc_01.jpg
  ...
  optera_doc_47.jpg
```

Then:

```bash
make run
```

Any directory works via `python3 run.py --input /path/to/images`.

Note that two files in the starter set are deliberately broken and the pipeline
is expected to handle both without a model call: `optera_doc_33.jpg` is a
Cloudflare HTML error page saved with a `.jpg` extension, and `optera_doc_47.jpg`
is a truncated JPEG.

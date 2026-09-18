# THE PARALLEL

A whole mind was cut into 166,691 pieces and published. One of them is yours.

**[Try it](https://nullsignal-hub.github.io/the-parallel/)** · [日本語版](https://nullsignal-hub.github.io/the-parallel/index.ja.html)

Answer ten questions — or skip them and just move your finger on a pad for six
seconds — and get matched to a real, individually identified neuron from a real
fly brain, fetched live from a public research dataset and rendered in your
browser from its actual traced 3D skeleton. Every number on the result screen
(synapse count, reach, population, branch points) is a measured value from the
data, not a generated one.

## What this is

- A single static page. No backend, no build step, no database. `index.html`
  and `index.ja.html` fetch skeleton data directly from a public Google Cloud
  Storage bucket and draw it on a `<canvas>`.
- Built on the [MaleCNS v1.0](https://www.malecns.org/) connectome dataset —
  166,691 traced neurons, released under CC BY 4.0 by the FlyEM Project Team
  at HHMI Janelia, the Cambridge Connectomics Group, and Google Research.
- The descriptive lines attached to each cell ("voice"/"line" in the data
  files) are written for this project; the measured statistics are not.

## Independence

Not affiliated with, endorsed by, or reviewed by HHMI, Janelia, the
University of Cambridge, MRC LMB, or Google. Cell identities and descriptions
here are illustrative and must not be used as a scientific reference.

## License

All rights reserved on the code, copy, and design in this repository unless
stated otherwise. The underlying MaleCNS skeleton and connectivity data is
separately licensed CC BY 4.0 by its original authors (see attribution
above) — that license covers the dataset, not this project's presentation
of it.

# Interface resources

Checked and applied on 12 October 2026. The package lock records the resolved dependency tree.

| Resource | Version / source | Actual use | Package notice |
| --- | --- | --- | --- |
| [Base UI](https://base-ui.com/react/overview/quick-start) | 1.9.0 | Select, Dialog, Tabs, Accordion, Tooltip and ToggleGroup/Toggle | MIT |
| [shadcn Base Nova Select](https://ui.shadcn.com/r/styles/base-nova/select.json) | Registry fetched 12 October 2026 | Select primitive/portal/list composition adapted in `web/src/ui.tsx`; original CSS replaces registry utilities and icons | MIT |
| [TanStack Table](https://tanstack.com/table/latest/docs/framework/react/react-table) | 8.21.3 | Typed rows, metric sorting, pagination and filtered-result state | MIT |
| [Observable Plot](https://observablehq.com/plot/features/marks) | 0.6.17 | Actual per-step loss/gradients and validation checkpoint plots | ISC |
| [Motion](https://motion.dev/docs/react) | 14.1.0 | Data-driven bar heights and aggregate metric transitions, respecting reduced motion | MIT |
| [Lucide](https://lucide.dev/) | 0.468.0 | Navigation, selected options, document expansion and status icons | ISC |
| [Barlow Condensed](https://fontsource.org/fonts/barlow-condensed) | Fontsource 5.2.7 | Condensed display typography and query-outcome counts | OFL-1.1 |
| [Manrope](https://fontsource.org/fonts/manrope) | Fontsource 5.2.8 | Reading, controls and numeric comparison | OFL-1.1 |

Both typefaces are self-hosted through npm, using Latin subsets. No remote font service, stock
photography, copied site assets or artificial three-dimensional decoration is needed for the
recorded experiment. The ranking mark and cover are original vectors.

The Select wrapper replaces the registry's Tailwind utilities and placeholder icons with the
project's CSS and Lucide. The popup anchors below the control, constrains itself to available
height and gives every option the same full width. Copied/adapted source is maintained locally;
registry updates do not replace it automatically.

Visual research included [Seesaw](https://www.seesaw.website/), the linked
[Moving Image Archive](https://www.movingimagearchive.com/) search/filter journey, and
[Bencho](https://bencho.dev/) interaction demonstrations. The applied lessons are content first,
clear next actions and local feedback tied to a real interaction. Their artwork, footage,
brand identities and implementation were not copied.

The locked toolchain uses Node 22.12+, Vite 7.3.7, TypeScript 5.9.3 and Vitest 5.0.3. Esbuild is
explicitly resolved to 0.28.2 after the earlier dependency audit; the current install/build was
checked with that version. React and React DOM are 19.2.0.

Benchmark attribution: [BEIR SciFact corpus and queries](https://huggingface.co/datasets/BeIR/scifact),
[relevance judgments](https://huggingface.co/datasets/BeIR/scifact-qrels), and
[MiniLM base encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Document
titles/abstracts and claim text in the explorer come from these public benchmark inputs.

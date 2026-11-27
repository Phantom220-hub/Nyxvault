# NYXVAULT training data sources

Last reviewed: 2026-11-27. The notebook records the actual Wikipedia page title, language, canonical URL, revision ID and retrieval date in `training_data/attributions.jsonl`.

| Source | License / conditions | Link | Date |
|---|---|---|---|
| NYXVAULT original `training/data/programming_seed.jsonl` | 103 original starter records: 89 legacy Q&A plus 14 structured programming tasks; authored for this project. | Repository file | 2026-11-26 |
| NYXVAULT original `training/data/code_projects_seed.jsonl` | 18 structured original programming tasks covering Python algorithms, data structures, error handling, Godot 4 gameplay systems, Kotlin architecture and a JavaScript Canvas game. All records use the `original` license label. | Repository file | 2026-11-27 |
| NYXVAULT `data/knowledge.jsonl` | Project-owned/maintainer-provided text; use only after confirming each entry was written by the project owner or otherwise licensed for reuse. | Repository file | 2026-11-02 |
| Russian Wikipedia | Generally CC BY-SA 4.0 and GFDL; page-level exceptions/imported text must be checked. Attribute page and contributors via page/history links; indicate modifications; share adaptations under compatible ShareAlike terms. | https://ru.wikipedia.org/ | 2026-11-02 |
| Tajik Wikipedia | Generally CC BY-SA 4.0 and GFDL; page-level exceptions/imported text must be checked. Same attribution and ShareAlike requirements. | https://tg.wikipedia.org/ | 2026-11-02 |
| Public-domain works | Include only after verifying public-domain status in the relevant jurisdictions and recording author, title, publication date and evidence. No public-domain corpus is bundled automatically. | Per-work record required | 2026-11-02 |
| User-authored essays and notes | Include only after the user confirms authorship or permission and agrees to their use for training. | User-supplied files | Pending user confirmation |

Wikipedia reuse guidance: https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse  
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/

The collection script searches programming and computer-science topics in Russian, Tajik and English, with a smaller selection of general science and education topics. The notebook also adds the 103-record starter seed set and the 18-record code-and-game-project seed set. It uses the official MediaWiki API and stores page-level attribution metadata. It does not download images or non-text media.

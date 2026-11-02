# NYXVAULT training data sources

Last reviewed: 2026-11-02. The notebook records the actual Wikipedia page title, language, canonical URL, revision ID and retrieval date in `training_data/attributions.jsonl`.

| Source | License / conditions | Link | Date |
|---|---|---|---|
| NYXVAULT `data/knowledge.jsonl` | Project-owned/maintainer-provided text; use only after confirming each entry was written by the project owner or otherwise licensed for reuse. | Repository file | 2026-11-02 |
| Russian Wikipedia | Generally CC BY-SA 4.0 and GFDL; page-level exceptions/imported text must be checked. Attribute page and contributors via page/history links; indicate modifications; share adaptations under compatible ShareAlike terms. | https://ru.wikipedia.org/ | 2026-11-02 |
| Tajik Wikipedia | Generally CC BY-SA 4.0 and GFDL; page-level exceptions/imported text must be checked. Same attribution and ShareAlike requirements. | https://tg.wikipedia.org/ | 2026-11-02 |
| Public-domain works | Include only after verifying public-domain status in the relevant jurisdictions and recording author, title, publication date and evidence. No public-domain corpus is bundled automatically. | Per-work record required | 2026-11-02 |
| User-authored essays and notes | Include only after the user confirms authorship or permission and agrees to their use for training. | User-supplied files | Pending user confirmation |

Wikipedia reuse guidance: https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse  
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/

The collection script searches technology/programming topics in Russian and Tajik, plus general Tajik essay topics (education, homeland, language, nature, science and youth). It uses the official MediaWiki API and stores page-level attribution metadata. It does not download images or non-text media.

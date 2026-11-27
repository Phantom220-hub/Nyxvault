# NYXVAULT raw training data

Upload new programming-training datasets into this folder.

## Expected file

Use a UTF-8 JSONL file, for example:

`nyxvault_game_training.jsonl`

Each line must be one complete JSON object. Do not wrap the file in Markdown fences and do not add comments between records.

## Required fields

Each record must follow `training/data_schema.py`:

- `id`: unique lowercase ID
- `level`: integer from 1 to 8
- `language`: supported language, e.g. `python`, `gdscript`, `javascript`
- `topic`: exact topic slug from `training/curriculum.py` for that level
- `type`: one of `explain`, `write_code`, `complete`, `debug`, `review`, `refactor`, `design`, `analyze`
- `instruction`: task description
- `output`: reference answer; put code inside fenced code blocks in the JSON string
- `license`: use `original` for newly written examples

Some task types also require `input`. Optional fields include `tests`, `tags`, and `source`.

## Important

The data preparation script scans this folder automatically:

`python training/prepare_data.py`

It validates structured JSONL records, removes duplicates, and writes the prepared train/validation files. Uploading a dataset does not itself train the model or change its weights.
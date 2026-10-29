# Database files

The Database task scores a predicted SQL query by executing it. The SQLite files are not in this repository. Download the test-suite databases from the Spider evaluation release, then place each database folder under `data/spider/databases/`.

1. Download the test suites (under 5 GB) from [this Google Drive file](https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view?usp=sharing). The same link is documented by [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval).
2. Unzip it. The archive contains a `database/` directory. Each subdirectory is one `db_id` and holds a `.sqlite` file.
3. Copy those subdirectories into `data/spider/databases/`.

After this step, `data/spider/databases/concert_singer/concert_singer.sqlite` should exist. The evaluator looks up `data/spider/databases/{db_id}/` using the `db_id` stored on each sample in `data/sharded_instructions_600.json`.

`pip install sqlparse nltk` is also required for this task.

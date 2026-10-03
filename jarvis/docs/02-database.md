# 02 — Database Layer

> The persistence layer for Jarvis notes and tasks. Pure Python, no ORM, no extra dependencies.

---

## Why SQLite?

SQLite is a file-based SQL database built into Python's standard library. You don't need to install anything. The entire database lives in a single `.db` file (`data/jarvis.db`).

For a local personal assistant this is ideal:
- Zero configuration — just a file path
- Full SQL — joins, filtering, ordering all work
- Portable — copy the file to back up your data
- Fast — more than fast enough for personal use

For JS developers: SQLite is like **lowdb or better-sqlite3** — file-based, embedded, no server needed. It's *not* like PostgreSQL or MySQL (which run as separate server processes).

---

## Schema

### `notes` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Unique identifier, auto-generated |
| `title` | TEXT | Short descriptive title |
| `content` | TEXT | The main body of the note |
| `tags` | TEXT | Tags stored as a JSON array string e.g. `["work", "ideas"]` |
| `created_at` | TEXT | ISO 8601 UTC timestamp e.g. `2026-10-03T14:30:00+00:00` |

### `tasks` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Unique identifier, auto-generated |
| `title` | TEXT | Short descriptive title |
| `description` | TEXT | Optional longer description |
| `status` | TEXT | `"pending"` or `"completed"` |
| `due_date` | TEXT | Optional date in `YYYY-MM-DD` format |
| `created_at` | TEXT | ISO 8601 UTC timestamp |

---

## Why string IDs (UUID)?

```python
from uuid import uuid4
note["id"] = str(uuid4())   # e.g. "f47ac10b-58cc-4372-a567-0e02b2c3d479"
```

Auto-incrementing integer IDs (`1`, `2`, `3`) are fine for a single database, but they cause collisions when you merge data, sync across devices, or export/import records. UUIDs are globally unique — same reason you'd use `crypto.randomUUID()` in JS instead of a counter.

---

## Why no ORM?

ORMs (like SQLAlchemy) add abstraction that hides what's happening. For learning purposes, seeing the raw SQL makes it clear exactly what each operation does. It's the same reason you might avoid Mongoose when learning MongoDB — start with the fundamentals.

```python
# Raw SQL — you can see exactly what's going into the database
self.conn.execute(
    "INSERT INTO notes (id, title, content, tags, created_at) VALUES (?, ?, ?, ?, ?)",
    (note["id"], note["title"], note["content"], json.dumps(note["tags"]), note["created_at"]),
)
```

The `?` placeholders are **parameterised queries** — the same pattern as `$1, $2` in PostgreSQL or `?` in better-sqlite3. They prevent SQL injection by keeping values separate from the SQL string.

---

## Why store tags as JSON?

SQLite doesn't have an array type. Options:
1. A separate `note_tags` junction table (normalised, more complex)
2. A comma-separated string `"work,ideas"` (simple but awkward to query)
3. **A JSON string** `'["work","ideas"]'` (simple, structured, queryable with `LIKE`)

We chose option 3. Tags are serialised with `json.dumps()` on write and `json.loads()` on read:

```python
# Writing
json.dumps(["work", "ideas"])   →   '["work", "ideas"]'

# Reading
json.loads('["work", "ideas"]') →   ["work", "ideas"]
```

---

## Using the Database class

```python
from db.database import Database

# Production — uses the path from config.py
db = Database()

# Or pass a path explicitly
db = Database("./my_data/jarvis.db")

# --- Notes ---
note = db.create_note(
    title="Meeting Notes",
    content="Discussed Q4 roadmap and hiring plans",
    tags=["work", "meetings"]
)
# → {"id": "...", "title": "Meeting Notes", "content": "...", "tags": [...], "created_at": "..."}

notes = db.list_notes(limit=10)       # newest first
note = db.get_note("some-uuid")       # or None
matches = db.search_notes_by_tag("work")

# --- Tasks ---
task = db.create_task(
    title="Write unit tests",
    description="Cover database and tool layers",
    due_date="2026-10-10"
)
# → {"id": "...", "title": "Write unit tests", "status": "pending", ...}

db.complete_task(task["id"])          # status → "completed"
pending = db.list_tasks(status="pending")
completed = db.list_tasks(status="completed")
db.delete_task(task["id"])            # → True
```

---

## Testing with `:memory:`

The `":memory:"` special path tells SQLite to create the database entirely in RAM.
It's destroyed when the `Database` object is garbage collected — perfect for tests.

```python
# In tests
db = Database(":memory:")   # fresh, empty, in-memory DB
```

This is the Python equivalent of using **Jest mocks** or an **in-memory store** — you get real database behaviour (real SQL, real constraints) without touching disk or leaving files behind.

Each test gets its own fresh `Database(":memory:")` via a pytest fixture:

```python
@pytest.fixture
def db():
    database = Database(":memory:")
    yield database      # test runs here
    database.close()    # cleanup after test
```

The `yield` keyword is what makes this a fixture that runs setup AND teardown — compare to Jest's `beforeEach`/`afterEach`:

```javascript
// Jest equivalent
let db;
beforeEach(() => { db = new InMemoryDatabase(); });
afterEach(() => { db.close(); });
```

---

## Running the tests

```bash
pytest tests/test_database.py -v
```

Expected output:
```
tests/test_database.py::TestNotes::test_create_note_returns_dict_with_all_fields PASSED
tests/test_database.py::TestNotes::test_create_note_with_tags PASSED
tests/test_database.py::TestNotes::test_get_note_returns_correct_note PASSED
...
tests/test_database.py::TestTasks::test_full_task_lifecycle PASSED

14 passed in 0.12s
```

# SQLite-based result store for Python simulations
#
# - Parameters: r, d, D (frequently changed) + q, g, f (also stored)
# - Supports multiple scalar results per run: energy, momentum, entropy, ...
# - Robust against float-key issues via quantized integer keys (EPS*)

import sqlite3
from spindyn.params import SpinParams
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

EPS = 1e-8
def key(x:float)->int : 
    return int(round(x/EPS))

## Database management Class
class DB:
    # Elementary helpers
    def __init__(self, path): # path : filename, xxx.sqlite
        self.path = path

    def _connect(self) :
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("PRAGMA journal_mode = WAL;")     # better for repeated writes
        con.execute("PRAGMA synchronous = NORMAL;")   # reasonable durability/speed tradeoff
        return con

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    # Schema helpers : Writing
    def init_db(self):
        con = self._connect()
        cur = con.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS param_sets (
            set_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            q REAL NOT NULL, g REAL NOT NULL, f REAL NOT NULL,
            kq INTEGER NOT NULL, kg INTEGER NOT NULL, kf INTEGER NOT NULL,
            UNIQUE (kq, kg, kf)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,

            r REAL NOT NULL, d REAL NOT NULL, DD REAL NOT NULL,
            kr INTEGER NOT NULL, kd INTEGER NOT NULL, kDD INTEGER NOT NULL,

            set_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
                    
            UNIQUE (kr, kd, kDD, set_id),
            FOREIGN KEY (set_id) REFERENCES param_sets(set_id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            run_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, name),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_r ON runs(r);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_d ON runs(d);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_DD ON runs(DD);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_set ON runs(set_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);")

        con.commit()
        con.close()

    def get_set_id(self, q, g, f):
        kq, kg, kf = key(q), key(g), key(f)

        con = self._connect()
        cur = con.cursor()

        cur.execute("""
        INSERT INTO param_sets (q,g,f,kq,kg,kf)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(kq,kg,kf) DO NOTHING
        """, (q, g, f, kq, kg, kf))

        cur.execute("SELECT set_id FROM param_sets WHERE kq=? AND kg=? AND kf=?", (kq, kg, kf))
        row = cur.fetchone()
        if row is None:
            con.close()
            raise RuntimeError("Failed to get set_id (unexpected).")

        set_id = int(row[0])
        con.commit()
        con.close()
        return set_id

    def upsert_run(self, param: SpinParams):
        q = param.q; g = param.g; f = param.f
        r = param.Omega/g; d = param.delta/g; D = param.D/g

        set_id = self.get_set_id(q, g, f)
        kr, kd, kD = key(r), key(d), key(D)
        now = self._now()

        con = self._connect()
        cur = con.cursor()

        cur.execute("""
        INSERT INTO runs (r,d,DD,kr,kd,kDD,set_id,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(kr,kd,kDD,set_id) DO NOTHING
        """, (r, d, D, kr, kd, kD, set_id, now))

        cur.execute("""
        SELECT run_id FROM runs WHERE kr=? AND kd=? AND kDD=? AND set_id=?
        """, (kr, kd, kD, set_id))
        row = cur.fetchone()
        if row is None:
            con.close()
            raise RuntimeError("Failed to get run_id (unexpected).")

        run_id = int(row[0])
        con.commit()
        con.close()
        return run_id

    def has_run(self, param: SpinParams) -> bool:
        q = param.q; g = param.g; f = param.f
        r = param.Omega/g; d = param.delta/g; D = param.D/g
        
        set_id = self.get_set_id(q, g, f)
        kr, kd, kD = key(r), key(d), key(D)

        con = self._connect()
        cur = con.cursor()
        cur.execute("SELECT 1 FROM runs WHERE kr=? AND kd=? AND kDD=? AND set_id=? LIMIT 1", (kr, kd, kD, set_id))
        ok = cur.fetchone() is not None
        con.close()
        return ok

    def upsert_metrics(self, run_id, metrics: Dict[str, float]):
        now = self._now()
        rows = [(run_id, str(k), float(v), now) for k, v in metrics.items()]

        con = self._connect()
        cur = con.cursor()
        cur.executemany("""
        INSERT INTO metrics (run_id, name, value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(run_id, name) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """, rows)
        con.commit()
        con.close()
    
    def has_metric(self, run_id, name) -> bool:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "SELECT 1 FROM metrics WHERE run_id=? AND name=? LIMIT 1",
            (int(run_id), str(name)),
        )
        ok = cur.fetchone() is not None
        con.close()
        return ok
    
    # Fetch helpers : Reading
    def metrics_by_id(self, run_id) -> Dict[str, float]:
        con = self._connect()
        cur = con.cursor()

        cur.execute("SELECT name, value FROM metrics WHERE run_id=? ORDER BY name", (run_id,))
        out = {name: float(val) for (name, val) in cur.fetchall()}
        con.close()
        return out

    def metrics_by_params(self, param: SpinParams) -> Dict[str, float]:
        run_id = self.get_run_id(param)
        if run_id is None:
            return {}
        return self.metrics_by_id(run_id)

    def get_run_id(self, param: SpinParams):
        q = param.q; g = param.g; f = param.f
        r = param.Omega/g; d = param.delta/g; D = param.D/g
        
        set_id = self.get_set_id(q, g, f)
        kr, kd, kD = key(r), key(d), key(D)

        con = self._connect()
        cur = con.cursor()
        cur.execute("SELECT run_id FROM runs WHERE kr=? AND kd=? AND kD=? AND set_id=? LIMIT 1", (kr, kd, kD, set_id))
        row = cur.fetchone()
        con.close()
        return int(row[0]) if row else None

    def list_results(
        self,
        metric_name: Optional[str] = None,
        r_range: Optional[Tuple[float, float]] = None,
        d_range: Optional[Tuple[float, float]] = None,
        D_range: Optional[Tuple[float, float]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
          {run_id, r,d,D,q,g,f, created_at, meta_json, metric_value?}
        """
        # construct where- SQL in where_sql
        where = []
        args: List[Any] = []

        if r_range is not None:
            where.append("t.r BETWEEN ? AND ?")
            args += [r_range[0], r_range[1]]
        if d_range is not None:
            where.append("t.d BETWEEN ? AND ?")
            args += [d_range[0], d_range[1]]
        if D_range is not None:
            where.append("t.DD BETWEEN ? AND ?")
            args += [D_range[0], D_range[1]]

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        con = self._connect()
        cur = con.cursor()
       
        # metric_name = None
        if metric_name is None:
            cur.execute(f"""
            SELECT t.run_id, t.r, t.d, t.DD, s.q, s.g, s.f, t.created_at
            FROM runs t
            JOIN param_sets s ON t.set_id = s.set_id
            {where_sql}
            ORDER BY t.run_id DESC
            LIMIT ?
            """, (*args, int(limit)))

            rows = cur.fetchall()
            con.close()
            return [
                {
                    "run_id": int(run_id),
                    "r": float(r), "d": float(d), "D": float(D),
                    "q": float(q), "g": float(g), "f": float(f),
                    "created_at": created_at,
                }
                for (run_id, r, d, D, q, g, f, created_at) in rows
            ]

        # metric name = not None
        cur.execute(f"""
        SELECT t.run_id, t.r, t.d, t.DD, s.q, s.g, s.f, m.updated_at, m.value
        FROM runs t
        JOIN param_sets s ON t.set_id = s.set_id
        LEFT JOIN metrics m ON (m.run_id = t.run_id AND m.name = ?)
        {where_sql}
        ORDER BY t.run_id DESC
        LIMIT ?
        """, (metric_name, *args, int(limit)))
        
        rows = cur.fetchall()
        con.close()
        return [
            {
                "run_id": int(run_id),
                "r": float(r), "d": float(d), "D": float(D),
                "q": float(q), "g": float(g), "f": float(f),
                "updated_at": updated_at,
                metric_name: (None if mv is None else float(mv)),
            }
            for (run_id, r, d, D, q, g, f, updated_at, mv) in rows
        ]

# Storage function
# run_id = store("results.sqlite", params, {"energy":..., "momentum":...})
def store(db: DB, param: SpinParams, metrics: Dict[str, float]):
    db.init_db()
    run_id = db.upsert_run(param)
    db.upsert_metrics(run_id, metrics)
    return run_id
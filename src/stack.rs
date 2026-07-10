//! The human-facing stack layer: a **named, inspectable priority
//! stack** over the task catalogue.
//!
//! Design notes (from the GID / OpenSoT comparison):
//!
//! - No `Begin/NextMotion/End` state machine (GID): a [`Stack`] is plain
//!   data — `Vec<Level>`, each level a list of `(name, Task)`. Level
//!   boundaries are code structure, not call timing, and an invalid
//!   call order is unrepresentable.
//! - Motion objectives are **explicit**: `println!("{stack}")` lists
//!   every level and task with its dimensions before solving, and
//!   [`Stack::solve`] returns a [`StackSolution`] whose
//!   [`report`](StackSolution::report) gives per-task achievement
//!   (equality residual, worst inequality margin) after solving — the
//!   two things GID's facade cannot show.
//! - Names flow into diagnostics: a degraded level is reported by name,
//!   not just index (OpenSoT's `task_id` idea).
//!
//! ```
//! # #[cfg(feature = "clarabel")] {
//! use misa_wbc::{Stack, SolveConfig, Task, VarLayout, tasks};
//! use nalgebra::{DMatrix, DVector};
//!
//! let vars = VarLayout::builder().add("x", 2).build();
//! let x = vars.var("x");
//! let stack = Stack::new(&vars)
//!     .level("pin", |l| {
//!         l.task(
//!             "x0",
//!             Task::equality(DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
//!                            DVector::from_vec(vec![1.0])),
//!         )
//!     })
//!     .level("rest", |l| l.task("reg", tasks::regularize(&x, &DVector::zeros(2))));
//!
//! println!("{stack}"); // the objectives, before solving
//! let sol = stack.solve(&SolveConfig::default()).unwrap();
//! println!("{}", sol.report()); // per-task achievement, after solving
//! assert!((sol.x[0] - 1.0).abs() < 1e-5);
//! # }
//! ```

use std::fmt;

use nalgebra::DVector;

use crate::affine::VarLayout;
use crate::solve::{solve_warm, Solution, SolveConfig, SolveStatus, WbcError};
use crate::task::Task;

/// One named task inside a level.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub task: Task,
}

/// One priority level: a name and its tasks (summed at solve time).
#[derive(Clone, Debug, Default)]
pub struct Level {
    pub name: String,
    pub entries: Vec<Entry>,
}

/// Builder for one level, passed to the closure of [`Stack::level`].
#[derive(Debug, Default)]
pub struct LevelBuilder {
    entries: Vec<Entry>,
}

impl LevelBuilder {
    /// Add a named task to this level. Any [`Task`] fits — from the
    /// [`crate::tasks`] catalogue, or hand-built.
    pub fn task(mut self, name: impl Into<String>, task: Task) -> Self {
        self.entries.push(Entry { name: name.into(), task });
        self
    }
}

/// A named, inspectable priority stack (index 0 = highest priority).
/// Plain data: build it, print it, solve it, keep it for the report.
#[derive(Clone, Debug)]
pub struct Stack {
    n_decision: usize,
    levels: Vec<Level>,
}

impl Stack {
    /// Start an empty stack over a variable layout (the layout fixes
    /// the decision-vector size every task must match).
    pub fn new(vars: &VarLayout) -> Self {
        Stack { n_decision: vars.n_decision(), levels: Vec::new() }
    }

    /// Append a priority level built by the closure. Levels are solved
    /// in the order they are added (first = highest priority).
    pub fn level(mut self, name: impl Into<String>, build: impl FnOnce(LevelBuilder) -> LevelBuilder) -> Self {
        let b = build(LevelBuilder::default());
        self.levels.push(Level { name: name.into(), entries: b.entries });
        self
    }

    pub fn n_decision(&self) -> usize {
        self.n_decision
    }

    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// The per-level summed tasks, as [`crate::solve::solve`] consumes them.
    pub fn to_tasks(&self) -> Vec<Task> {
        self.levels
            .iter()
            .map(|l| {
                l.entries
                    .iter()
                    .map(|e| e.task.clone())
                    .fold(Task::empty(self.n_decision), |acc, t| acc + t)
            })
            .collect()
    }

    /// Solve the stack. The returned [`StackSolution`] keeps a copy of
    /// the stack so [`StackSolution::report`] can attribute achievement
    /// per named task.
    pub fn solve(&self, cfg: &SolveConfig) -> Result<StackSolution, WbcError> {
        self.solve_warm(cfg, None)
    }

    /// [`Stack::solve`] with a warm anchor (see [`crate::solve::solve_warm`]).
    pub fn solve_warm(
        &self,
        cfg: &SolveConfig,
        warm_anchor: Option<&DVector<f64>>,
    ) -> Result<StackSolution, WbcError> {
        let solution = solve_warm(&self.to_tasks(), cfg, warm_anchor)?;
        Ok(StackSolution { stack: self.clone(), solution })
    }
}

impl fmt::Display for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "stack ({} decision vars, {} levels)", self.n_decision, self.levels.len())?;
        for (i, l) in self.levels.iter().enumerate() {
            writeln!(f, "  level {i} \"{}\"", l.name)?;
            for e in &l.entries {
                writeln!(
                    f,
                    "    {:<24} eq {:>3}  iq {:>3}",
                    e.name,
                    e.task.n_eq(),
                    e.task.n_iq()
                )?;
            }
        }
        Ok(())
    }
}

/// A solved stack: the raw [`Solution`] plus the stack it came from,
/// so achievement can be reported per named task.
#[derive(Clone, Debug)]
pub struct StackSolution {
    stack: Stack,
    solution: Solution,
}

impl StackSolution {
    pub fn status(&self) -> &SolveStatus {
        &self.solution.status
    }

    /// The raw solution (decision vector, warm anchor).
    pub fn solution(&self) -> &Solution {
        &self.solution
    }

    /// Per-task achievement at the solution: for each named task its
    /// equality residual `‖A·x − b‖` and its worst inequality margin
    /// `min(f − D·x)` (negative = violated). This is the explicit
    /// "did each motion objective get met, and by how much" view.
    pub fn report(&self) -> Report {
        let x = &self.solution.x;
        let levels = self
            .stack
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let degraded = matches!(
                    self.solution.status,
                    SolveStatus::Degraded { level, .. } if level == i
                );
                let tasks = l
                    .entries
                    .iter()
                    .map(|e| {
                        let eq_residual = (e.task.n_eq() > 0)
                            .then(|| (&e.task.a * x - &e.task.b).norm());
                        let iq_margin = (e.task.n_iq() > 0).then(|| {
                            let m = &e.task.f - &e.task.d * x;
                            m.min()
                        });
                        TaskReport { name: e.name.clone(), eq_residual, iq_margin }
                    })
                    .collect();
                LevelReport { name: l.name.clone(), degraded, tasks }
            })
            .collect();
        Report { levels }
    }
}

impl std::ops::Deref for StackSolution {
    type Target = Solution;
    fn deref(&self) -> &Solution {
        &self.solution
    }
}

/// Achievement of one named task at the solution.
#[derive(Clone, Debug)]
pub struct TaskReport {
    pub name: String,
    /// `‖A·x − b‖` (None if the task has no equality rows).
    pub eq_residual: Option<f64>,
    /// `min(f − D·x)` — the tightest inequality margin; negative means
    /// violated (None if the task has no inequality rows).
    pub iq_margin: Option<f64>,
}

/// Achievement of one level.
#[derive(Clone, Debug)]
pub struct LevelReport {
    pub name: String,
    /// Whether this level was the (first) degraded one.
    pub degraded: bool,
    pub tasks: Vec<TaskReport>,
}

/// The full per-task achievement report; human-readable via `Display`.
#[derive(Clone, Debug)]
pub struct Report {
    pub levels: Vec<LevelReport>,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, l) in self.levels.iter().enumerate() {
            let mark = if l.degraded { "DEGRADED" } else { "optimal" };
            writeln!(f, "level {i} \"{}\": {mark}", l.name)?;
            for t in &l.tasks {
                write!(f, "  {:<24}", t.name)?;
                if let Some(r) = t.eq_residual {
                    write!(f, " ‖A·x−b‖ = {r:9.3e}")?;
                }
                if let Some(m) = t.iq_margin {
                    write!(f, "  margin min = {m:+9.3e}")?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "clarabel"))]
mod tests {
    use super::*;
    use crate::tasks;
    use nalgebra::{DMatrix, DVector};

    fn two_level() -> (VarLayout, Stack) {
        let vars = VarLayout::builder().add("x", 2).build();
        let x = vars.var("x");
        let stack = Stack::new(&vars)
            .level("pin", |l| {
                l.task(
                    "x0=1",
                    Task::equality(
                        DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
                        DVector::from_vec(vec![1.0]),
                    ),
                )
                .task(
                    "x1≤3",
                    Task::inequality(
                        DMatrix::from_row_slice(1, 2, &[0.0, 1.0]),
                        DVector::from_vec(vec![3.0]),
                    ),
                )
            })
            .level("rest", |l| l.task("reg", tasks::regularize(&x, &DVector::zeros(2))));
        (vars, stack)
    }

    #[test]
    fn display_lists_objectives() {
        let (_, stack) = two_level();
        let s = format!("{stack}");
        assert!(s.contains("level 0 \"pin\""), "{s}");
        assert!(s.contains("x0=1"), "{s}");
        assert!(s.contains("level 1 \"rest\""), "{s}");
    }

    #[test]
    fn solve_and_report_achievement() {
        let (_, stack) = two_level();
        let sol = stack.solve(&SolveConfig::default()).unwrap();
        assert_eq!(*sol.status(), SolveStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-5);

        let report = sol.report();
        // The pinned equality is achieved…
        let pin = &report.levels[0].tasks[0];
        assert!(pin.eq_residual.unwrap() < 1e-5, "pin residual {:?}", pin.eq_residual);
        // …and the inequality has non-negative margin.
        let iq = &report.levels[0].tasks[1];
        assert!(iq.iq_margin.unwrap() >= -1e-9, "margin {:?}", iq.iq_margin);
        // Human-readable rendering carries the names.
        let text = format!("{report}");
        assert!(text.contains("x0=1") && text.contains("margin"), "{text}");
    }

    #[test]
    fn stack_matches_raw_solve() {
        let (_, stack) = two_level();
        let via_stack = stack.solve(&SolveConfig::default()).unwrap();
        let raw = crate::solve::solve(&stack.to_tasks(), &SolveConfig::default()).unwrap();
        assert!((&via_stack.x - &raw.x).norm() < 1e-12, "stack must be a pure veneer");
    }
}

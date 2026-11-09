// An immutable state store with a reducer (original): a todo/project tracker
// where every action produces a new state via object spread, Object.assign,
// array map/filter/concat/slice and computed keys. Selectors with memoization
// derive views; a structural-sharing check counts how much each action copies.
const ACTIONS = typeof RD_ACTIONS === 'number' ? RD_ACTIONS : 8000;

const initial = Object.freeze({
  users: {},
  projects: {},
  tasks: {},
  order: [],
  filter: { status: 'all', owner: null },
  history: [],
  version: 0,
});

function reducer(state, action) {
  switch (action.type) {
    case 'addUser':
      return { ...state, users: { ...state.users, [action.id]: { id: action.id, name: action.name, load: 0 } } };
    case 'addProject':
      return { ...state, projects: { ...state.projects, [action.id]: { id: action.id, title: action.title, taskIds: [] } } };
    case 'addTask': {
      const task = { id: action.id, project: action.project, owner: action.owner, title: action.title, done: false, points: action.points, labels: [] };
      const proj = state.projects[action.project];
      const owner = state.users[action.owner];
      return {
        ...state,
        tasks: { ...state.tasks, [task.id]: task },
        projects: { ...state.projects, [proj.id]: { ...proj, taskIds: [...proj.taskIds, task.id] } },
        users: { ...state.users, [owner.id]: { ...owner, load: owner.load + task.points } },
        order: state.order.concat(task.id),
      };
    }
    case 'toggle': {
      const t = state.tasks[action.id];
      if (!t) return state;
      const owner = state.users[t.owner];
      const delta = t.done ? t.points : -t.points;
      return Object.assign({}, state, {
        tasks: Object.assign({}, state.tasks, { [t.id]: { ...t, done: !t.done } }),
        users: { ...state.users, [owner.id]: { ...owner, load: owner.load + delta } },
      });
    }
    case 'label': {
      const t = state.tasks[action.id];
      if (!t || t.labels.includes(action.label)) return state;
      return { ...state, tasks: { ...state.tasks, [t.id]: { ...t, labels: [...t.labels, action.label].sort() } } };
    }
    case 'reassign': {
      const t = state.tasks[action.id];
      if (!t || t.owner === action.owner) return state;
      const from = state.users[t.owner], to = state.users[action.owner];
      const pts = t.done ? 0 : t.points;
      return {
        ...state,
        tasks: { ...state.tasks, [t.id]: { ...t, owner: to.id } },
        users: { ...state.users, [from.id]: { ...from, load: from.load - pts }, [to.id]: { ...to, load: to.load + pts } },
      };
    }
    case 'remove': {
      const t = state.tasks[action.id];
      if (!t) return state;
      const { [t.id]: _gone, ...rest } = state.tasks;
      const proj = state.projects[t.project];
      return {
        ...state,
        tasks: rest,
        projects: { ...state.projects, [proj.id]: { ...proj, taskIds: proj.taskIds.filter((x) => x !== t.id) } },
        order: state.order.filter((x) => x !== t.id),
      };
    }
    case 'filter':
      return { ...state, filter: { ...state.filter, ...action.filter } };
    default:
      return state;
  }
}

function withHistory(inner) {
  return (state, action) => {
    const next = inner(state, action);
    if (next === state) return state;
    const history = state.history.length >= 20 ? state.history.slice(1) : state.history.slice();
    history.push(action.type);
    return { ...next, history, version: state.version + 1 };
  };
}

function memo1(fn) {
  let lastArg = undefined, lastResult = undefined, hits = 0;
  const f = (arg) => { if (arg === lastArg) { hits++; return lastResult; } lastArg = arg; lastResult = fn(arg); return lastResult; };
  f.hits = () => hits;
  return f;
}
const selectOpenPoints = memo1((tasks) => Object.values(tasks).filter((t) => !t.done).reduce((s, t) => s + t.points, 0));
const selectBusiest = memo1((users) => Object.values(users).reduce((best, u) => (best === null || u.load > best.load ? u : best), null));

let seed = 77;
function r(n) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed >>> 8) % n; }

const reduce = withHistory(reducer);
let state = initial;
for (let u = 0; u < 12; u++) state = reduce(state, { type: 'addUser', id: 'u' + u, name: 'User ' + u });
for (let p = 0; p < 6; p++) state = reduce(state, { type: 'addProject', id: 'p' + p, title: 'Project ' + p });
let nextTask = 0;
let checks = 0;
const LABELS = ['bug', 'ui', 'perf', 'docs', 'infra'];
for (let i = 0; i < ACTIONS; i++) {
  const k = r(100);
  const live = state.order.length;
  let action;
  if (k < 25 || live < 20) action = { type: 'addTask', id: 't' + nextTask++, project: 'p' + r(6), owner: 'u' + r(12), title: 'Task ' + i, points: 1 + r(8) };
  else if (k < 55) action = { type: 'toggle', id: state.order[r(live)] };
  else if (k < 70) action = { type: 'label', id: state.order[r(live)], label: LABELS[r(5)] };
  else if (k < 82) action = { type: 'reassign', id: state.order[r(live)], owner: 'u' + r(12) };
  else if (k < 97 || live > 300) action = { type: 'remove', id: state.order[r(live)] };
  else action = { type: 'filter', filter: { status: ['all', 'open', 'done'][r(3)] } };
  if (live > 150 && action.type === 'addTask') action = { type: 'remove', id: state.order[r(live)] };
  const prev = state;
  state = reduce(state, action);
  if (state !== prev && state.users === prev.users) checks++;
  if (i % 10 === 0) { selectOpenPoints(state.tasks); selectBusiest(state.users); }
}
const busiest = selectBusiest(state.users);
const loads = Object.values(state.users).map((u) => u.load).join(',');
const labelled = Object.values(state.tasks).filter((t) => t.labels.length > 1).length;
console.log('version=' + state.version, 'tasks=' + state.order.length, 'created=' + nextTask, 'open=' + selectOpenPoints(state.tasks), 'busiest=' + busiest.id, 'shared=' + checks);
console.log('loads=' + loads, 'labelled=' + labelled, 'history=' + state.history.slice(-5).join('/'), 'memo=' + selectOpenPoints.hits() + '/' + selectBusiest.hits());

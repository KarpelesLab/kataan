import { Kataan } from './kataan.js';
import { EXAMPLES } from './examples.js';

const TIMEOUT_MS = 5000;
const engine = new Kataan({ timeoutMs: TIMEOUT_MS });

// The pipeline, in the order the engine walks it. `verb` is what the stage did,
// so the timing line reads as a sentence.
const STAGES = [
  { id: 'run', label: 'Value', verb: 'ran' },
  { id: 'lex', label: 'Tokens', verb: 'tokenized' },
  { id: 'parse', label: 'Tree', verb: 'parsed' },
];

// Figures come from the nightly gate; the workflow rewrites them at build time
// so the page cannot quietly drift from the ledger.
const CONFORMANCE = {
  rate: '99.45%',
  passing: '51,603',
  ran: '51,890',
  ledger: 287,
  clusters: [
    { name: 'intl402/Temporal', count: 27 },
    { name: 'intl402/NumberFormat', count: 27 },
    { name: 'built-ins/Atomics', count: 21 },
    { name: 'intl402/DateTimeFormat', count: 18 },
    { name: 'built-ins/Array', count: 18 },
  ],
};
CONFORMANCE.worst = Math.max(...CONFORMANCE.clusters.map((c) => c.count));

const WAYS = [
  {
    title: 'Rust',
    code: 'let (printed, value) = kataan::eval_source("6 * 7")?;',
    note: 'A library with no build script and no C toolchain.',
  },
  {
    title: 'C',
    code: 'size_t n = sizeof out;\nkt_eval("6 * 7", 5, out, &n);',
    note: 'A stable ABI with an in/out length convention. What this page calls.',
  },
  {
    title: 'CLI',
    code: '$ kataan run -e \'6 * 7\'\n$ kataan lex -e \'x => x\'',
    note: 'Run a script, or dump any stage of the pipeline.',
  },
];

const { createApp } = window.Vue;

createApp({
  data() {
    return {
      examples: EXAMPLES,
      stages: STAGES,
      conformance: CONFORMANCE,
      ways: WAYS,
      build: {
        version: '0.0.8',
        gzip: '2.4 MB',
        intlShare: 'two thirds',
        timeout: `${TIMEOUT_MS / 1000}-second`,
      },
      source: EXAMPLES[0].source,
      activeExample: EXAMPLES[0].id,
      note: EXAMPLES[0].note,
      stage: 'run',
      // Results are kept per stage so switching panes does not re-run a script
      // that may have printed something.
      results: {},
      busy: false,
      rerun: false,
      booted: false,
      bootError: '',
      stale: false,
      modKey: navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl+',
    };
  },

  computed: {
    result() {
      return this.results[this.stage] ?? null;
    },
    stageVerb() {
      return STAGES.find((s) => s.id === this.stage).verb;
    },
  },

  watch: {
    // Any edit invalidates every stage's output: showing a tree that no longer
    // matches the source would be worse than showing nothing.
    source() {
      this.stale = true;
    },
  },

  async mounted() {
    try {
      await engine.ready;
      this.booted = true;
      this.run();
    } catch (error) {
      this.bootError = `The engine failed to load: ${error.message}`;
    }
  },

  methods: {
    load(example) {
      this.source = example.source;
      this.activeExample = example.id;
      this.note = example.note;
      this.results = {};
      this.stale = false;
      this.run();
    },

    select(stage) {
      this.stage = stage;
      if (!this.results[stage] || this.stale) this.run();
    },

    /** Tab indents rather than leaving the editor — this is a code field. */
    indent(event) {
      const el = event.target;
      const { selectionStart: from, selectionEnd: to } = el;
      this.source = `${this.source.slice(0, from)}  ${this.source.slice(to)}`;
      this.$nextTick(() => {
        el.selectionStart = el.selectionEnd = from + 2;
      });
    },

    async run() {
      if (!this.booted) return;
      // A stage switch while a run is in flight must not be dropped on the
      // floor — remember it and honour it once the current run returns.
      if (this.busy) {
        this.rerun = true;
        return;
      }
      this.busy = true;
      const stage = this.stage;
      const wasStale = this.stale;
      try {
        const result = await engine.run(this.source, stage);
        this.results = wasStale ? { [stage]: result } : { ...this.results, [stage]: result };
        this.stale = false;
      } finally {
        this.busy = false;
        if (this.rerun) {
          this.rerun = false;
          this.run();
        }
      }
    },
  },
}).mount('#app');

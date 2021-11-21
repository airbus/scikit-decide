<template>
  <span class="nav-item" v-if="options && options.length > 0">
    Version:
    <select v-model="selected" @change="onChange">
      <option v-for="option in options" :value="option.value">
        {{ option.text }}
      </option>
    </select>
  </span>
</template>

<script>
import Axios from "axios";
export default {
  data() {
    return {
      selected: undefined,
      options: [],
    };
  },
  created: async function () {
    try {
      let res = await Axios.get(
        "https://api.github.com/repos/airbus/scikit-decide/git/trees/gh-pages"
      );
      const versionNode = res.data.tree.find((e) => {
        return e.path.toLowerCase() === "version";
      });
      res = await Axios.get(versionNode.url);
      this.options = res.data.tree.map((e) => {
        return { value: e.path, text: e.path };
      });
      this.options.sort((e1, e2) => {
        const e1Arr = e1.text.split(".");
        const e2Arr = e2.text.split(".");
        for (let i = 0; i < e1Arr.length && i < e2Arr.length; i++) {
          const e1V = parseInt(e1Arr[i]);
          const e2V = parseInt(e2Arr[i]);
          if (e1V !== e2V) return e2V - e1V;
          if (e1Arr[i] !== e2Arr[i]) return e2Arr[i] - e1Arr[i];
        }
        return e1.text === e2.text ? 0 : e2.text < e1.text ? -1 : 1;
      });
      this.options.unshift({ value: "master", text: "dev" });
      const path = window.location.pathname.toLowerCase();
      if (path.startsWith("/scikit-decide/version/")) {
        const start = 23; // len("/version/scikit-decide/")
        const end = path.indexOf("/", start);
        this.selected = path.substring(start, end);
      } else {
        this.selected = "master";
      }
    } catch (ex) {}
  },
  methods: {
    onChange(event) {
      const targetVersionPath =
        this.selected === "master" ? "" : `/version/${this.selected}`;
      const path = window.location.pathname.toLowerCase();
      let startIdx = 14; // len("/scikit-decide")
      const versionIdx = path.indexOf("/version/");
      if (versionIdx >= 0) {
        startIdx = versionIdx + 9; // len("/version/")
      }

      const endIdx = path.indexOf("/", startIdx);

      window.location.pathname =
        window.location.pathname.substring(0, 14) +
        targetVersionPath +
        window.location.pathname.substring(endIdx);
    },
  },
};
</script>

<script>
export default {
  name: "App",
  data() {
    return {
      message: "Hello Vue!",
      infinitySpeak: "",
      pixiApp: "",
    };
  },
  async mounted() {
    const cubism2Model =
      "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/shizuku/shizuku.model.json";
    // const cubism4Model =
    //   "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/haru/haru_greeter_t03.model3.json";

    const live2d = PIXI.live2d;

    const pixiApp = new PIXI.Application({
      view: document.getElementById("canvas"),
      autoStart: true,
      resizeTo: window,
      backgroundColor: 0x333333,
    });

    this.pixiApp = pixiApp;

    const models = await Promise.all([
      live2d.Live2DModel.from(cubism2Model),
      // live2d.Live2DModel.from(cubism4Model),
    ]);

    models.forEach((model) => {
      pixiApp.stage.addChild(model);

      const scaleX = (innerWidth * 0.3) / model.width;
      const scaleY = (innerHeight * 0.5) / model.height;

      // fit the window
      model.scale.set(Math.min(scaleX, scaleY));

      model.y = innerHeight * 0.1;
    });

    const model2 = models[0];

    let soundCounter = 0;
    let infinitySpeak = (del) => {
      // console.log(model2.internalModel.motionManager);
      const animationGroup =
        model2.internalModel.motionManager.state.currentGroup;
      if (animationGroup == "idle") {
        model2.motion("flick_head");
      }
    };
    this.infinitySpeak = infinitySpeak;
    pixiApp.ticker.add(infinitySpeak);

    model2.on("hit", (hitAreas) => {
      if (hitAreas.includes("body")) {
        console.log(model2.internalModel.motionManager);
        model2.internalModel.motionManager.startMotion("flick_head", 0);
      }
    });

    function draggable(model) {
      model.buttonMode = true;
      model.on("pointerdown", (e) => {
        model.dragging = true;
        model._pointerX = e.data.global.x - model.x;
        model._pointerY = e.data.global.y - model.y;
      });
      model.on("pointermove", (e) => {
        if (model.dragging) {
          model.position.x = e.data.global.x - model._pointerX;
          model.position.y = e.data.global.y - model._pointerY;
        }
      });
      model.on("pointerupoutside", () => (model.dragging = false));
      model.on("pointerup", () => (model.dragging = false));
    }
  },
  methods: {
    stopSpeak() {
      this.pixiApp.ticker.remove(this.infinitySpeak);
    },
  },
};
</script>

<template>
  <button @click="stopSpeak">Stop</button>
  <canvas id="canvas"></canvas>
  <div id="control"></div>
</template>

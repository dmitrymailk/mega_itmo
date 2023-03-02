<script>
export default {
  name: "App",
  data() {
    return {
      infinitySpeak: "",
      pixiApp: "",
      chat: [
        {
          type: "bot",
          message: "Hi, how are you?",
        },
        {
          type: "user",
          message: "Hi",
        },
      ],
    };
  },
  async mounted() {
    const cubism2Model =
      "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/shizuku/shizuku.model.json";
    // const cubism4Model =
    //   "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/haru/haru_greeter_t03.model3.json";

    const live2d = PIXI.live2d;
    const container = window.document.querySelector(".my-app");

    const pixiApp = new PIXI.Application({
      view: document.getElementById("canvas"),
      autoStart: true,
      // resizeTo: window,
      backgroundColor: 0x333333,
      width: window.innerWidth,
      height: 2000,
      antialias: true,
    });

    this.pixiApp = pixiApp;

    const models = await Promise.all([
      live2d.Live2DModel.from(cubism2Model),
      // live2d.Live2DModel.from(cubism4Model),
    ]);

    models.forEach((model) => {
      pixiApp.stage.addChild(model);

      const scaleX = (innerWidth * 1) / model.width;
      const scaleY = (innerHeight * 1) / model.height;

      // // fit the window
      model.scale.set(Math.min(scaleX, scaleY));

      model.y = innerHeight * 0.4;
      model.x = innerHeight * 0.4;
    });

    const model2 = models[0];

    let infinitySpeak = (del) => {
      // console.log(model2.internalModel.motionManager);
      const animationGroup =
        model2.internalModel.motionManager.state.currentGroup;
      if (animationGroup == "idle") {
        model2.motion("flick_head");
      }
    };
    this.infinitySpeak = infinitySpeak;
    // pixiApp.ticker.add(infinitySpeak);

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

    window.Webcam.set({
      width: 320,
      height: 240,
      image_format: "jpeg",
      jpeg_quality: 90,
    });
    window.Webcam.attach("#my_camera");
  },
  methods: {
    stopSpeak() {
      this.pixiApp.ticker.remove(this.infinitySpeak);
    },
    take_snapshot() {
      // take snapshot and get image data
      window.Webcam.snap(function (data_uri) {
        // display results in page
        console.log("data_uri");
        window
          .axios({
            method: "post",
            url: "http://127.0.0.1:5000/analyse-mood/",
            data: {
              image: data_uri,
            },
          })
          .then((e) => {
            console.log(e);
          });
      });
    },
  },
};
</script>

<template>
  <div class="flex h-screen">
    <div class="flex flex-row h-full w-full overflow-x-hidden">
      <div class="flex flex-col h-full w-1/2 bg-white my-app overflow-y-hidden">
        <canvas id="canvas"></canvas>
      </div>
      <div class="flex flex-col flex-auto w-1/2 h-full">
        <div
          class="flex flex-col flex-auto flex-shrink-0 rounded-2xl bg-gray-100 h-full p-4"
        >
          <div class="flex flex-col h-full overflow-x-auto mb-4">
            <div class="flex flex-col h-full">
              <div class="grid grid-cols-12 gap-y-2">
                <template v-for="(item, index) in chat" :key="index">
                  <template v-if="item['type'] == 'bot'">
                    <div class="col-start-1 col-end-8 p-3 rounded-lg">
                      <div class="flex flex-row items-center">
                        <div
                          class="flex items-center justify-center h-10 w-10 rounded-full bg-indigo-500 flex-shrink-0"
                        >
                          BOT
                        </div>
                        <div
                          class="relative ml-3 text-sm bg-white py-2 px-4 shadow rounded-xl"
                        >
                          <div>{{ item["message"] }}</div>
                        </div>
                      </div>
                    </div>
                  </template>
                  <template v-else>
                    <div class="col-start-6 col-end-13 p-3 rounded-lg">
                      <div
                        class="flex items-center justify-start flex-row-reverse"
                      >
                        <div
                          class="flex items-center justify-center h-10 w-10 rounded-full bg-indigo-500 flex-shrink-0"
                        >
                          YOU
                        </div>
                        <div
                          class="relative mr-3 text-sm bg-indigo-100 py-2 px-4 shadow rounded-xl"
                        >
                          <div>{{ item["message"] }}</div>
                        </div>
                      </div>
                    </div>
                  </template>
                </template>
              </div>
            </div>
          </div>
          <div
            class="flex flex-row items-center h-16 rounded-xl bg-white w-full px-4"
          >
            <div>
              <button
                class="flex items-center justify-center text-gray-400 hover:text-gray-600"
              >
                <svg
                  class="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                  ></path>
                </svg>
              </button>
            </div>
            <div class="flex-grow ml-4">
              <div class="relative w-full">
                <input
                  type="text"
                  class="flex w-full border rounded-xl focus:outline-none focus:border-indigo-300 pl-4 h-10"
                />
                <button
                  @click="take_snapshot"
                  class="absolute flex items-center justify-center h-full w-12 right-0 top-0 text-gray-400 hover:text-gray-600"
                >
                  <svg
                    class="w-6 h-6"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    ></path>
                  </svg>
                </button>
              </div>
            </div>
            <div class="ml-4">
              <button
                class="flex items-center justify-center bg-indigo-500 hover:bg-indigo-600 rounded-xl text-white px-4 py-1 flex-shrink-0"
              >
                <span>Send</span>
                <span class="ml-2">
                  <svg
                    class="w-4 h-4 transform rotate-45 -mt-px"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    ></path>
                  </svg>
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="hidden">
    <div id="my_camera"></div>
  </div>
</template>

<style>
#my_camera {
  width: 32px;
  height: 32px;
  border: 1px solid black;
}
</style>

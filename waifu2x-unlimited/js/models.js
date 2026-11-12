const MODELS = [
    {
        arch: "swin_unet_v3",
        calc_tile_size: (tile_size) => {
            if (tile_size % 32 != 0) {
                tile_size = tile_size + (32 - tile_size % 32);
            }
            tile_size = Math.max(tile_size, 64);
            return tile_size;
        },
        styles: {
            art: { color_stability: true, padding: "replication" },
        },
        methods: [
            { name: "scale1x", scale: 1, offset: 8, has_noise: true },
            { name: "scale2x", scale: 2, offset: 16, has_noise: true },
        ]
    },
    {
        arch: "swin_unet",
        calc_tile_size: (tile_size) => {
            while (true) {
                if ((tile_size - 16) % 12 === 0 && (tile_size - 16) % 16 === 0) {
                    break;
                }
                tile_size += 1;
            }
            return tile_size;
        },
        styles: {
            art: { color_stability: true, padding: "replication" },
            art_scan: { color_stability: false, padding: "replication" },
            photo: { color_stability: false, padding: "reflection" }
        },
        methods: [
            { name: "scale1x", scale: 1, offset: 8, has_noise: true },
            { name: "scale2x", scale: 2, offset: 16, has_noise: true },
            { name: "scale4x", scale: 4, offset: 32, has_noise: true }
        ]
    },
    {
        arch: "cunet",
        calc_tile_size: (tile_size, config) => {
            const adj = config.scale === 1 ? 16 : 32;
            tile_size = ((tile_size * config.scale + config.offset * 2) - adj) / config.scale;
            tile_size -= tile_size % 4;
            return tile_size;
        },
        styles: {
            art: { color_stability: true, padding: "replication" }
        },
        methods: [
            { name: "scale1x", scale: 1, offset: 28, has_noise: true },
            { name: "scale2x", scale: 2, offset: 36, has_noise: true }
        ]
    }
];

function gen_arch_config() {
    const config = {};

    for (const model of MODELS) {
        config[model.arch] = {};
        for (const [style, style_config] of Object.entries(model.styles)) {
            config[model.arch][style] = {};
            for (const m of model.methods) {
                const base_config = {
                    arch: model.arch,
                    domain: style,
                    calc_tile_size: model.calc_tile_size,
                    ...style_config,
                    scale: m.scale,
                    offset: m.offset
                };
                config[model.arch][style][m.name] = base_config;
                if (m.has_noise) {
                    for (let i = 0; i < 4; i++) {
                        const noise_method = (m.name === "scale1x") ? `noise${i}` : `noise${i}_${m.name}`;
                        config[model.arch][style][noise_method] = base_config;
                    }
                }
            }
        }
    }

    return config;
}

export const CONFIG = {
    arch: gen_arch_config(),
    model_base_url: "models/",
    get_model_base_url: function () {
        return this.model_base_url;
    },
    set_model_base_url: function (baseUrl) {
        let value = (baseUrl || "").trim() || "models/";

        if (/^https?:\/\//i.test(value)) {
            const url = new URL(value);
            if (url.search || url.hash) {
                throw new Error("Model base URL must not include a query string or fragment.");
            }
            if (!url.pathname.endsWith("/")) {
                url.pathname += "/";
            }
            value = url.href;
        } else {
            if (/^[a-z][a-z0-9+.-]*:/i.test(value) || value.startsWith("//")) {
                throw new Error("Use an HTTP(S) URL or a relative folder path such as models/.");
            }
            value = value.endsWith("/") ? value : value + "/";
        }

        this.model_base_url = value;
        return value;
    },
    resolve_model_path: function (path) {
        if (/^https?:\/\//i.test(this.model_base_url)) {
            return new URL(path, this.model_base_url).href;
        }
        return this.model_base_url + path;
    },
    get_config: function (arch, style, method) {
        if ((arch in this.arch) && (style in this.arch[arch]) && (method in this.arch[arch][style])) {
            const config = { ...this.arch[arch][style][method] };
            config["path"] = this.resolve_model_path(`${arch}/${style}/${method}.onnx`);
            return config;
        } else {
            return null;
        }
    },
    get_helper_model_path: function (name) {
        return this.resolve_model_path(`utils/${name}.onnx`);
    }
};

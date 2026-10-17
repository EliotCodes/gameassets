# Wren — sky-pirate aviator

A fully procedural, human-proportioned character (about 1.78 m) built with Blender's Python API.
Nothing is hand-sculpted: `build_aviator.py` generates the whole model from code.

- **Body and hands**: Skin-modifier armature graphs. The hands have five fingers with three segments each, plus nails.
- **Head**: a dense sphere shaped by analytic features (nose, brow, sockets, lips, chin, sculpted eyelids). It also has
  eyeballs with an iris shader and a refractive cornea, lashes, ears, eyebrows made of individual hairs, and freckles.
- **Hair**: about 8k clumped, tapered curve strands that follow a flow field over the skull.
- **Clothing**: the jacket, shirt and pants are made by copying the body surface, trimming it and inflating it
  along the normals, so they always fit. On top of that: a shearling collar, a scarf, lace-up boots with eyelets,
  bracers, a belt and pouches, a satchel, and brass goggles. Straps and props are snapped onto surfaces with BVH ray casts.

```
blender -b --python build_aviator.py -- --render out.png --view full|portrait|face|back|side \
        --samples 128 --res 1080x1440 --save wren.blend
```

Renders are in `renders/`. `wren_aviator.blend` is the generated scene.

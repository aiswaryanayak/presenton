# servers/fastapi/services/pptx_presentation_creator.py
import os
import uuid
from typing import List, Optional

from lxml import etree
from lxml.etree import fromstring, tostring
from PIL import Image

from pptx import Presentation
from pptx.shapes.autoshape import Shape
from pptx.slide import Slide
from pptx.text.text import _Paragraph, TextFrame, Font, _Run
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt
from pptx.dml.color import RGBColor

from services.html_to_text_runs_service import (
    parse_html_text_to_text_runs as parse_inline_html_to_runs
)

from utils.download_helpers import download_files
from utils.image_utils import (
    clip_image,
    create_circle_image,
    fit_image,
    invert_image,
    round_image_corners,
    set_image_opacity,
)

from models.pptx_models import (
    PptxAutoShapeBoxModel,
    PptxBoxShapeEnum,
    PptxConnectorModel,
    PptxFillModel,
    PptxFontModel,
    PptxParagraphModel,
    PptxPictureBoxModel,
    PptxPositionModel,
    PptxPresentationModel,
    PptxShadowModel,
    PptxSlideModel,
    PptxSpacingModel,
    PptxStrokeModel,
    PptxTextBoxModel,
    PptxTextRunModel,
)


BLANK_SLIDE_LAYOUT = 6


class PptxPresentationCreator:
    def __init__(self, ppt_model: PptxPresentationModel, temp_dir: str):
        self._ppt_model = ppt_model
        self._slide_models = ppt_model.slides or []
        self._temp_dir = temp_dir

        self._ppt = Presentation()
        self._ppt.slide_width = Pt(1280)
        self._ppt.slide_height = Pt(720)

    # --------------------------
    # Helper XML Element Builder
    # --------------------------
    def get_sub_element(self, parent, tagname, **kwargs):
        el = OxmlElement(tagname)
        for k, v in kwargs.items():
            el.set(k, str(v))
        parent.append(el)
        return el

    # --------------------------
    # Fetch Remote Images First
    # --------------------------
    async def fetch_network_assets(self):
        image_urls = []
        models = []

        # global shapes
        for shape in self._ppt_model.shapes or []:
            if isinstance(shape, PptxPictureBoxModel):
                path = shape.picture.path
                if isinstance(path, str) and path.startswith("http"):
                    image_urls.append(path)
                    models.append(shape)

        # slide shapes
        for slide in self._slide_models:
            for shape in slide.shapes or []:
                if isinstance(shape, PptxPictureBoxModel):
                    path = shape.picture.path
                    if isinstance(path, str) and path.startswith("http"):
                        image_urls.append(path)
                        models.append(shape)

        if not image_urls:
            return

        try:
            downloaded = await download_files(image_urls, self._temp_dir)
        except Exception as e:
            print("❌ download_files failed:", e)
            downloaded = [None] * len(image_urls)

        for model, path in zip(models, downloaded):
            if path:
                model.picture.path = path
                model.picture.is_network = False
            else:
                model.picture.path = "/static/images/placeholder.jpg"

    # --------------------------
    # Create PPT (Main Entry)
    # --------------------------
    async def create_ppt(self):
        await self.fetch_network_assets()

        for slide_model in self._slide_models:
            global_shapes = list(self._ppt_model.shapes or [])
            local_shapes = list(slide_model.shapes or [])

            merged_shapes = global_shapes + local_shapes

            slide_model.shapes = merged_shapes
            self.add_and_populate_slide(slide_model)

    # --------------------------
    # Add Slide + All Shapes
    # --------------------------
    def add_and_populate_slide(self, slide_model: PptxSlideModel):
        try:
            slide = self._ppt.slides.add_slide(self._ppt.slide_layouts[BLANK_SLIDE_LAYOUT])
        except Exception:
            slide = self._ppt.slides.add_slide(self._ppt.slide_layouts[0])

        # Background
        if slide_model.background:
            try:
                self.apply_fill_to_shape(slide.background, slide_model.background)
            except Exception as e:
                print("⚠️ background fill failed:", e)

        # Notes
        if slide_model.note:
            try:
                slide.notes_slide.notes_text_frame.text = slide_model.note
            except:
                pass

        # Shapes
        for shape_model in slide_model.shapes:
            try:
                if isinstance(shape_model, PptxPictureBoxModel):
                    self.add_picture(slide, shape_model)
                elif isinstance(shape_model, PptxAutoShapeBoxModel):
                    self.add_autoshape(slide, shape_model)
                elif isinstance(shape_model, PptxTextBoxModel):
                    self.add_textbox(slide, shape_model)
                elif isinstance(shape_model, PptxConnectorModel):
                    self.add_connector(slide, shape_model)
                else:
                    self.add_textbox(
                        slide,
                        PptxTextBoxModel(
                            position=PptxPositionModel(left=50, top=50, width=300, height=80),
                            paragraphs=[PptxParagraphModel(text=str(shape_model))]
                        )
                    )
            except Exception as e:
                print("⚠️ shape failed:", e)

    # --------------------------
    # Picture
    # --------------------------
    def add_picture(self, slide: Slide, model: PptxPictureBoxModel):
        path = model.picture.path
        if not path:
            return

        try:
            image = Image.open(path).convert("RGBA")

            # Border radius (supports list or int)
            if model.border_radius:
                radius = model.border_radius[0] if isinstance(model.border_radius, list) else model.border_radius
                image = round_image_corners(image, radius)

            # Object-fit
            if model.object_fit and model.object_fit.fit:
                fit_mode = model.object_fit.fit.value
                image = fit_image(image, model.position.width, model.position.height, fit_mode)

            elif model.clip:
                image = clip_image(image, model.position.width, model.position.height)

            # Circle shape
            if model.shape == PptxBoxShapeEnum.CIRCLE:
                image = create_circle_image(image)

            # Invert
            if model.invert:
                image = invert_image(image)

            # Opacity
            if model.opacity is not None:
                image = set_image_opacity(image, model.opacity)

            # Save transformed image
            out = os.path.join(self._temp_dir, f"{uuid.uuid4()}.png")
            image.save(out)
            path = out

        except Exception as e:
            print("⚠️ image transform failed:", e)
            path = "/static/images/placeholder.jpg"

        pos = self.get_margined_position(model.position, model.margin)
        slide.shapes.add_picture(path, *pos.to_pt_list())

    # --------------------------
    # Autoshape
    # --------------------------
    def add_autoshape(self, slide: Slide, model: PptxAutoShapeBoxModel):
        try:
            pos = self.get_margined_position(model.position, model.margin)
            shape = slide.shapes.add_shape(model.type, *pos.to_pt_list())

            tf = shape.text_frame
            tf.word_wrap = model.text_wrap

            self.apply_fill_to_shape(shape, model.fill)
            self.apply_stroke_to_shape(shape, model.stroke)
            self.apply_shadow_to_shape(shape, model.shadow)
            self.apply_border_radius_to_shape(shape, model.border_radius)

            if model.paragraphs:
                self.add_paragraphs(tf, model.paragraphs)

        except Exception as e:
            print("⚠️ autoshape failed:", e)

    # --------------------------
    # Textbox
    # --------------------------
    def add_textbox(self, slide: Slide, model: PptxTextBoxModel):
        try:
            pos = self.get_margined_position(model.position, model.margin)
            box = slide.shapes.add_textbox(*pos.to_pt_list())
            tf = box.text_frame
            tf.word_wrap = model.text_wrap

            self.apply_fill_to_shape(box, model.fill)
            self.apply_margin_to_text_box(tf, model.margin)
            self.add_paragraphs(tf, model.paragraphs)

        except Exception as e:
            print("⚠️ textbox failed:", e)

    # --------------------------
    # Connector
    # --------------------------
    def add_connector(self, slide: Slide, model: PptxConnectorModel):
        try:
            conn = slide.shapes.add_connector(model.type, *model.position.to_pt_xyxy())
            conn.line.width = Pt(model.thickness)
            conn.line.color.rgb = RGBColor.from_string(model.color)

            # FIXED: Apply opacity on line fill
            self.set_fill_opacity(conn.line.fill, model.opacity)

        except Exception as e:
            print("⚠️ connector failed:", e)

    # --------------------------
    # Paragraph Handler
    # --------------------------
    def add_paragraphs(self, tf: TextFrame, paragraphs: List[PptxParagraphModel]):
        for idx, p_model in enumerate(paragraphs):
            p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            self.populate_paragraph(p, p_model)

    def populate_paragraph(self, p: _Paragraph, model: PptxParagraphModel):
        try:
            if model.spacing:
                self.apply_spacing_to_paragraph(p, model.spacing)

            if model.line_height:
                p.line_spacing = model.line_height

            if model.alignment:
                p.alignment = model.alignment

            if model.font:
                self.apply_font(p.font, model.font)

            runs = []
            if model.text:
                runs = self.parse_html_text_to_text_runs(model.font, model.text)
            elif model.text_runs:
                runs = model.text_runs

            for run in runs:
                r = p.add_run()
                self.populate_text_run(r, run)

        except Exception as e:
            print("⚠️ populate paragraph failed:", e)

    def parse_html_text_to_text_runs(self, font, text):
        return parse_inline_html_to_runs(text, font)

    def populate_text_run(self, r: _Run, model: PptxTextRunModel):
        try:
            r.text = model.text
            if model.font:
                self.apply_font(r.font, model.font)
        except Exception as e:
            print("⚠️ run failed:", e)

    # --------------------------
    # Attribute Helpers
    # --------------------------
    def apply_border_radius_to_shape(self, shape: Shape, radius: Optional[int]):
        if not radius:
            return
        try:
            adj = Pt(radius) / max(min(shape.width, shape.height), Pt(1))
            shape.adjustments[0] = adj
        except Exception:
            pass

    def apply_fill_to_shape(self, shape: Shape, fill: Optional[PptxFillModel]):
        if not fill:
            shape.fill.background()
            return
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill.color)
        self.set_fill_opacity(shape.fill, fill.opacity)

    def apply_stroke_to_shape(self, shape: Shape, stroke: Optional[PptxStrokeModel]):
        if not stroke or stroke.thickness == 0:
            shape.line.fill.background()
            return
        shape.line.fill.solid()
        shape.line.fill.fore_color.rgb = RGBColor.from_string(stroke.color)
        shape.line.width = Pt(stroke.thickness)
        self.set_fill_opacity(shape.line.fill, stroke.opacity)

    def apply_shadow_to_shape(self, shape: Shape, shadow: Optional[PptxShadowModel]):
        try:
            sp = shape._element
            spPr = sp.xpath("p:spPr")[0]
            ns = spPr.nsmap

            effect_list = spPr.find("a:effectLst", namespaces=ns)
            if effect_list is None:
                effect_list = etree.SubElement(spPr, f"{{{ns['a']}}}effectLst")

            old = effect_list.find("a:outerShdw", namespaces=ns)
            if old is not None:
                effect_list.remove(old)

            if shadow is None:
                sh = etree.SubElement(effect_list, f"{{{ns['a']}}}outerShdw", {"blurRad": "0", "dist": "0", "dir": "0"})
                col = etree.SubElement(sh, f"{{{ns['a']}}}srgbClr", {"val": "000000"})
                etree.SubElement(col, f"{{{ns['a']}}}alpha", {"val": "0"})
                return

            blur = str(int(Pt(shadow.radius)))
            dist = str(int(Pt(shadow.offset)))
            dir_val = str(int(shadow.angle * 1000))

            sh = etree.SubElement(
                effect_list,
                f"{{{ns['a']}}}outerShdw",
                {"blurRad": blur, "dist": dist, "dir": dir_val}
            )

            col = etree.SubElement(sh, f"{{{ns['a']}}}srgbClr", {"val": shadow.color})
            alpha = str(int(shadow.opacity * 100000))
            etree.SubElement(col, f"{{{ns['a']}}}alpha", {"val": alpha})

        except Exception as e:
            print("⚠️ shadow failed:", e)

    def set_fill_opacity(self, fill, opacity):
        if opacity is None or opacity >= 1:
            return
        try:
            alpha = int(opacity * 100000)
            ts = fill._xPr.solidFill if hasattr(fill._xPr, "solidFill") else None
            if ts:
                self.get_sub_element(ts, "a:alpha", val=str(alpha))
        except Exception:
            pass

    def get_margined_position(self, pos: PptxPositionModel, margin: Optional[PptxSpacingModel]):
        if not margin:
            return pos
        return PptxPositionModel(
            left=pos.left + margin.left,
            top=pos.top + margin.top,
            width=max(pos.width - margin.left - margin.right, 0),
            height=max(pos.height - margin.top - margin.bottom, 0),
        )

    # --------------------------
    # FINAL SAVE
    # --------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self._ppt.save(path)
        except Exception as e:
            print("❌ save failed:", e)
            raise


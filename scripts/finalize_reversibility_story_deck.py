"""Add presentation actions and export the clickable PDF backup."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"p": P, "a": A, "r": R, "pr": PR}


def xml(root):
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def powerpoint(build, output, links, count):
    descriptions = json.loads((build / "plot-descriptions.json").read_text())
    with ZipFile(build / "source.pptx") as deck:
        files = {filename: deck.read(filename) for filename in deck.namelist()}
    for index in range(1, count + 1):
        s = f"ppt/slides/slide{index}.xml"
        r = f"ppt/slides/_rels/slide{index}.xml.rels"
        slide = etree.fromstring(files[s])
        rels = etree.fromstring(files[r])
        if str(index) in descriptions:
            (picture,) = slide.xpath(".//p:pic/p:nvPicPr/p:cNvPr", namespaces=NS)
            picture.set("name", f"Slide {index} research chart")
            picture.set("descr", descriptions[str(index)])
        seen = {rel.get("Id") for rel in rels}
        for number, link in enumerate((item for item in links if item["slide"] == index), 1):
            properties = slide.xpath(".//p:cNvPr[@name=$name]", namespaces=NS, name=link["name"])
            if len(properties) != 1:
                raise ValueError(f"Link {link['name']} matched {len(properties)} elements on slide {index}")
            rid = f"rIdADNN{number}"
            if rid in seen:
                raise ValueError("Hyperlink relationship already exists")
            click = etree.SubElement(properties[0], f"{{{A}}}hlinkClick", {f"{{{R}}}id": rid})
            target = link["target"]
            relation = {"Id": rid}
            if isinstance(target, int):
                assert 1 <= target <= count
                click.set("action", "ppaction://hlinksldjump")
                relation.update(Type=R + "/slide", Target=f"slide{target}.xml")
            else:
                assert target.startswith("https://")
                relation.update(Type=R + "/hyperlink", Target=target, TargetMode="External")
            etree.SubElement(rels, f"{{{PR}}}Relationship", relation)
        transition = etree.Element(f"{{{P}}}transition", spd="fast")
        etree.SubElement(transition, f"{{{P}}}fade")
        before = next((child for child in slide if etree.QName(child).localname in ("timing", "extLst")), None)
        if before is None:
            slide.append(transition)
        else:
            slide.insert(slide.index(before), transition)
        files[s] = xml(slide)
        files[r] = xml(rels)
    with ZipFile(output, "w", compression=8, compresslevel=6) as updated:
        for filename, contents in files.items():
            updated.writestr(filename, contents)
    with ZipFile(output) as finished:
        assert finished.testzip() is None
        for index in range(1, count + 1):
            slide = etree.fromstring(finished.read(f"ppt/slides/slide{index}.xml"))
            assert slide.xpath("count(p:transition/p:fade)", namespaces=NS) == 1
            assert slide.xpath("count(.//a:hlinkClick)", namespaces=NS) == sum(i["slide"] == index for i in links)
            if str(index) in descriptions:
                assert slide.xpath("string(.//p:pic/p:nvPicPr/p:cNvPr/@descr)", namespaces=NS) == descriptions[str(index)]


def backup_pdf(build, output, links, count):
    canvas = Canvas(str(output), pagesize=(960, 540), pageCompression=1)
    canvas.setTitle("Measuring Action Reversibility | ADNN")
    for index in range(1, count + 1):
        canvas.bookmarkPage(f"slide-{index}")
        canvas.drawImage(ImageReader(build / f"slide-{index:02d}.png"), 0, 0, 960, 540)
        for item in (link for link in links if link["slide"] == index):
            x, y, w, h = (item[name] * .75 for name in ("x", "y", "w", "h"))
            rect = (x, 540 - y - h, x + w, 540 - y)
            if isinstance(item["target"], int):
                canvas.linkRect("", f"slide-{item['target']}", rect, relative=0, thickness=0)
            else:
                canvas.linkURL(item["target"], rect, relative=0, thickness=0)
        canvas.showPage()
    canvas.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    links = json.loads((args.build / "navigation.json").read_text())
    count = len(list(args.build.glob("slide-*.png")))
    assert count == 20
    powerpoint(args.build, args.output / "reversibility-talk.pptx", links, count)
    backup_pdf(args.build, args.output / "reversibility-talk.pdf", links, count)
    print(f"{count} slides, {len(links)} verified navigation actions, outline fade and clickable PDF backup.")


if __name__ == "__main__":
    main()

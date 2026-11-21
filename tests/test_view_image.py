"""Synthetic PNG/JPEG tests for evaluator-backed multimodal image handling."""
import asyncio
import base64
import json
import struct
import unittest
import zlib
from unittest.mock import patch
import serve

from serve import Agent, marked, response_input, MAX_IMAGE_BYTES


def png(width=8, height=8):
    def chunk(name, payload):
        return (
            struct.pack(">I", len(payload)) + name + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xffffffff)
        )
    scanline = b"\x00" + b"\xff\x00\x00" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanline * height))
        + chunk(b"IEND", b"")
    )


def jpeg(width=8, height=8):
    # Minimal synthetic JPEG SOF0 metadata for parser tests.
    segment = (
        b"\x08" + struct.pack(">HH", height, width)
        + b"\x01\x01\x11\x00"
    )
    return (
        b"\xff\xd8\xff\xc0"
        + struct.pack(">H", 2 + len(segment))
        + segment
        + b"\xff\xd9"
    )


class Session:
    workdir = "/app"
    mcp_servers = []
    instruction = "Inspect the supplied synthetic drawing"
    limits = {}

    def __init__(self, data=None, remaining=900):
        self.files = {"/app/input/plan.png": png()} if data is None else data
        self.read_calls = []
        self.remaining = remaining
        self.final_message = ""

    def time_left(self):
        return self.remaining

    async def read_file(self, path):
        self.read_calls.append(path)
        if path not in self.files:
            raise RuntimeError("missing synthetic test file")
        return self.files[path]

    async def log(self, message):
        pass


class ViewImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_png_attaches_native_image_without_text_base64(self):
        session = Session()
        agent = Agent(session)
        output = await agent.run_tool("view_image", {"path": "input/plan.png"})
        self.assertIn("image attached", output)
        self.assertEqual(session.read_calls, ["/app/input/plan.png"])
        self.assertEqual(len(agent.pending_images), 1)
        msg = agent.pending_images[0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"][1]["type"], "text")
        media = msg["content"][0]
        self.assertEqual(media["type"], "image_url")
        self.assertEqual(media["image_url"]["detail"], "high")
        prefix, encoded = media["image_url"]["url"].split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(encoded), png())
        self.assertNotIn(encoded, output)

    async def test_jpeg_attaches_right_mime(self):
        agent = Agent(Session({"/app/figure.jpg": jpeg()}))
        out = await agent.view_image("figure.jpg")
        self.assertIn("image/jpeg", out)
        self.assertIn("image/jpeg", agent.pending_images[0]["content"][0]["image_url"]["url"])

    async def test_jpeg_invalid_dimensions_rejected(self):
        agent = Agent(Session({"/app/oversized.jpg": jpeg(10000, 8)}))
        self.assertIn("too-large JPEG", await agent.view_image("oversized.jpg"))
        self.assertEqual(agent.pending_images, [])

    async def test_jpeg_missing_frame_rejected(self):
        agent = Agent(Session({"/app/notreal.jpg": b"\xff\xd8\xff\xd9"}))
        self.assertIn("missing or malformed JPEG", await agent.view_image("notreal.jpg"))
        self.assertFalse(agent.pending_images)

    async def test_jpeg_with_trailing_metadata_accepted(self):
        agent = Agent(Session({"/app/meta.jpg": jpeg() + b"\x00\x00"}))
        self.assertIn("image/jpeg", await agent.view_image("meta.jpg"))

    async def test_cropped_png_magnifies_and_preserves_pixel_color(self):
        from PIL import Image
        from io import BytesIO
        agent = Agent(Session())
        response = await agent.view_image(
            "input/plan.png", crop=[1, 1, 7, 7], scale=3
        )
        self.assertIn("crop=[1, 1, 7, 7]", response)
        encoded = agent.pending_images[0]["content"][0]["image_url"]["url"].split(",")[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.size, (18, 18))
            self.assertEqual(image.convert("RGB").getpixel((9, 9)), (255, 0, 0))

    async def test_invalid_crop_never_attaches(self):
        agent = Agent(Session())
        for crop in ([0, 0, 900, 900], [-1, 1, 5, 5],
                     [4, 4, 2, 7], [True, 0, 4, 4]):
            with self.subTest(crop=crop):
                self.assertIn("error:", await agent.view_image(
                    "input/plan.png", crop=crop, scale=2
                ))
        self.assertEqual(agent.pending_images, [])

    async def test_repeated_same_crop_avoids_duplicate(self):
        agent = Agent(Session())
        self.assertIn("image attached", await agent.view_image(
            "input/plan.png", crop=[0, 0, 5, 5], scale=2
        ))
        self.assertIn("already attached", await agent.view_image(
            "input/plan.png", crop=[0, 0, 5, 5], scale=2
        ))
        self.assertEqual(len(agent.pending_images), 1)

    async def test_scale_without_crop_is_rejected(self):
        agent = Agent(Session())
        result = await agent.view_image("input/plan.png", scale=2)
        self.assertIn("crop must be", result)
        self.assertFalse(agent.pending_images)

    async def test_crop_of_actual_jpeg_decodes_to_png(self):
        from PIL import Image
        from io import BytesIO
        jpeg = BytesIO()
        Image.new("RGB", (32, 16), (0, 20, 250)).save(jpeg, format="JPEG")
        agent = Agent(Session({"/app/chart.jpg": jpeg.getvalue()}))
        result = await agent.view_image(
            "chart.jpg", crop=[4, 3, 28, 14], scale=2
        )
        self.assertIn("image/png", result)
        uri = agent.pending_images[0]["content"][0]["image_url"]["url"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    async def test_near_deadline_rejects_image_with_summary_margin(self):
        session = Session(remaining=170)
        agent = Agent(session)
        agent.finish_margin = 160
        self.assertIn("too close", await agent.view_image("input/plan.png"))
        self.assertFalse(session.read_calls)

    async def test_unsupported_and_binary_files_do_not_attach(self):
        files = {
            "/app/thing.bin": b"random bytes",
            "/app/bad.png": b"\x89PNG\r\n\x1a\n" + b"x" * 50,
        }
        agent = Agent(Session(files))
        self.assertIn("not a PNG or JPEG", await agent.view_image("thing.bin"))
        self.assertIn("incomplete PNG", await agent.view_image("bad.png"))
        self.assertEqual(agent.pending_images, [])

    async def test_rejects_oversize(self):
        agent = Agent(Session({"/app/huge.png": b"\x00" * (MAX_IMAGE_BYTES + 1)}))
        out = await agent.view_image("huge.png")
        self.assertIn("1 to", out)
        self.assertFalse(agent.pending_images)

    async def test_png_dimension_bomb_rejected(self):
        tiny = bytearray(png())
        tiny[16:24] = struct.pack(">II", 10000, 10000)
        agent = Agent(Session({"/app/big.png": bytes(tiny)}))
        out = await agent.view_image("big.png")
        self.assertIn("too-large PNG dimensions", out)
        self.assertFalse(agent.pending_images)

    async def test_unresponsive_task_image_rpc_times_out_and_recovers(self):
        class HungSession(Session):
            async def read_file(self, path):
                self.read_calls.append(path)
                await asyncio.sleep(120)
                return png()
        session = HungSession()
        agent = Agent(session)
        with patch.object(serve, "MAX_IMAGE_READ_SECONDS", 0.02):
            response = await agent.view_image("input/plan.png")
        self.assertIn("read timed out", response)
        self.assertIn("another task tool", response)
        self.assertFalse(agent.pending_images)
        self.assertEqual(len(session.read_calls), 1)

    async def test_too_close_to_deadline_never_reads(self):
        session = Session(remaining=1)
        agent = Agent(session)
        agent.finish_margin = 1
        self.assertIn("too close", await agent.view_image("input/plan.png"))
        self.assertEqual(session.read_calls, [])

    async def test_marked_and_responses_conversion(self):
        agent = Agent(Session())
        await agent.view_image("input/plan.png")
        msg = agent.pending_images[0]
        messages = [{"role": "system", "content": "SYSTEM"}, msg]
        with patch.object(serve, "CACHE_MARK", True):
            cache = marked(messages)
        self.assertEqual(cache[1]["content"][0]["type"], "image_url")
        self.assertEqual(
            cache[1]["content"][1]["cache_control"], {"type": "ephemeral"}
        )
        self.assertNotIn("cache_control", messages[1]["content"][1])
        response = response_input(messages)
        self.assertEqual(response[1]["content"][0]["type"], "input_image")
        self.assertEqual(response[1]["content"][1]["type"], "input_text")
        self.assertEqual(response[1]["content"][0]["detail"], "high")
        self.assertTrue(response[1]["content"][0]["image_url"].startswith("data:image/png;base64,"))

    async def test_duplicate_image_is_not_appended_twice(self):
        agent = Agent(Session())
        first = await agent.view_image("input/plan.png")
        second = await agent.view_image("input/plan.png")
        self.assertIn("image attached", first)
        self.assertIn("already attached", second)
        self.assertEqual(len(agent.pending_images), 1)

    async def test_truncated_png_is_rejected(self):
        agent = Agent(Session({"/app/truncated.png": png()[:-5]}))
        self.assertIn("incomplete PNG", await agent.view_image("truncated.png"))
        self.assertEqual(agent.pending_images, [])

    async def test_text_token_estimate_is_not_base64_byte_count(self):
        agent = Agent(Session())
        await agent.view_image("input/plan.png")
        agent.messages = [
            {"role": "system", "content": "SYSTEM"},
            agent.pending_images[0],
        ]
        estimate = agent.estimate_prompt_tokens()
        self.assertLess(estimate, 3000)
        self.assertGreater(estimate, 1500)

    async def test_invalid_image_endpoint_can_strip_media_safely(self):
        agent = Agent(Session())
        await agent.view_image("input/plan.png")
        agent.messages.append(agent.pending_images.pop())
        self.assertTrue(agent.drop_images())
        self.assertFalse(agent.drop_images())
        self.assertTrue(all(
            not isinstance(m.get("content"), list)
            for m in agent.messages
        ))
        self.assertNotIn(
            "view_image",
            [t.get("function", {}).get("name") for t in agent.tools],
        )
        self.assertEqual(agent.image_bytes, 0)

    async def test_recovery_remedy_drops_rejected_image(self):
        agent = Agent(Session())
        await agent.view_image("input/plan.png")
        agent.messages.append(agent.pending_images.pop())
        self.assertTrue(await agent.remedy("Unsupported image_url content"))
        self.assertNotIn("image_url", str(agent.messages))

    async def test_agent_run_delivers_user_image_after_tool_result(self):
        session = Session()
        agent = Agent(session)

        async def no_mcp():
            return ""

        agent.connect_mcp = no_mcp
        class Reply:
            def __init__(self, name=None):
                self.text = "DONE" if name is None else ""
                self.finish = "stop" if name is None else "tool_calls"
                self.calls = (
                    {0: {"id": "call_1", "name": name,
                          "arguments": json.dumps({"path": "input/plan.png"})}}
                    if name else {}
                )
            def message(self):
                if not self.calls:
                    return {"role": "assistant", "content": self.text}
                c = next(iter(self.calls.values()))
                return {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": c["id"], "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]}
                    }],
                }
        rounds = 0
        async def respond(**kwargs):
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                return Reply("view_image")
            self.assertTrue(any(
                m["role"] == "user" and isinstance(m["content"], list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for m in agent.messages
            ))
            self.assertTrue(any(m["role"] == "tool" for m in agent.messages))
            return Reply()
        agent.complete = respond
        output = await agent.run()
        self.assertEqual(output, "DONE")
        self.assertEqual(rounds, 2)
        self.assertEqual(agent.pending_images, [])


if __name__ == "__main__":
    unittest.main()

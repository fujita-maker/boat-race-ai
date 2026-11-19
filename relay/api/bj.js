// v229: boatrace.jp 中継（Vercel・東京リージョン hnd1）
// Render(海外)からは boatrace.jp が 403 で拒否されるため、ここ(東京)で代わりに読んで HTML をそのまま返す。
// 使い方: GET /api/bj?p=/owpc/pc/race/odds3t?rno=1%26jcd=04%26hd=20261119&key=合言葉
module.exports = async (req, res) => {
  const key = process.env.RELAY_KEY || "";
  const q = req.query || {};
  if (!key || q.key !== key) { res.status(403).send("bad key"); return; }
  const p = String(q.p || "");
  if (!/^\/owpc\/pc\/race\/[a-z0-9]+\?[A-Za-z0-9=&_%-]+$/.test(p)) { res.status(400).send("bad path"); return; }
  try {
    const r = await fetch("https://www.boatrace.jp" + p, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ja,en;q=0.8",
      },
      signal: AbortSignal.timeout(18000),
    });
    const body = await r.text();
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.status(r.status).send(body);
  } catch (e) {
    res.status(502).send("relay error: " + String(e).slice(0, 200));
  }
};

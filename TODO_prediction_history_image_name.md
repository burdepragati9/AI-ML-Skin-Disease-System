# TODO - Prediction History: Image Name column + image viewing

## Step 1: Locate rendering logic
- [x] Identify Prediction History UI in `app.py` (`render_prediction_history`).
- [x] Confirm it uses `recent_searches()` which returns rows from `searches` table.

## Step 2: Identify stored image path availability
- [x] Verify `recent_searches()` returns full row including `image_path` (from `SELECT * FROM searches`).

## Step 3: Implement Image Name column in UI (no logic changes)
- [ ] Add a new `Image Name` column to the dataframe output for Prediction History search results.
- [ ] The Image Name should be a clickable link.
- [ ] Clicking should open/show the corresponding image using the stored `image_path`.

## Step 4: Display rules
- [ ] Keep all existing filters, pagination, and table behavior.
- [ ] Do not modify prediction/ML logic, DB schema, auth, or any other functionality.

## Step 5: Test
- [ ] Run the app and validate:
  - Table shows Image Name column.
  - Link opens or previews the correct uploaded image for each row.

